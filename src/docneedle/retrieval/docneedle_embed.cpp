#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <string>
#include <mutex>

struct docneedle_embed_ctx {
    llama_model * model;
    llama_context * lctx;
    mtmd_context * mctx;
    int dimension;
    std::mutex mtx;
};

// Use hardware sqrtss directly via __builtin_sqrtf to prevent dynamic linking
// against glibc 2.43 libm sqrtf symbol, guaranteeing full portability to HF ZeroGPU.
static void normalize_l2(float * vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = __builtin_sqrtf(norm);
    if (norm > 0.0f) {
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < dim; i++) {
            vec[i] *= inv_norm;
        }
    }
}

extern "C" {

void * docneedle_embed_init(
    const char * model_path,
    const char * mmproj_path,
    int n_gpu_layers,
    int n_threads,
    int dim
) {
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 2048;
    int actual_threads = (n_threads > 0 && n_threads <= 64) ? n_threads : 4;
    cparams.n_threads = actual_threads;
    cparams.n_threads_batch = actual_threads;

    // Bridge struct ABI shift between header (160 bytes, offset 128) and
    // runtime libllama.so (168 bytes, offset 136). Setting both offsets ensures
    // embeddings are enabled regardless of llama.cpp version.
    unsigned char cparams_buf[256];
    std::memset(cparams_buf, 0, sizeof(cparams_buf));
    std::memcpy(cparams_buf, &cparams, sizeof(cparams));
    cparams_buf[128] = 1;
    cparams_buf[136] = 1;

    llama_context * lctx = llama_init_from_model(model, *(llama_context_params*)cparams_buf);
    if (!lctx) {
        llama_model_free(model);
        return nullptr;
    }
    llama_set_embeddings(lctx, true);

    mtmd_context_params mcparams = mtmd_context_params_default();
    mcparams.use_gpu = (n_gpu_layers != 0);
    mcparams.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mcparams);
    if (!mctx) {
        llama_free(lctx);
        llama_model_free(model);
        return nullptr;
    }

    auto * ctx = new docneedle_embed_ctx();
    ctx->model = model;
    ctx->lctx = lctx;
    ctx->mctx = mctx;
    ctx->dimension = (dim > 0) ? dim : 768;
    return ctx;
}

int docneedle_embed_text(
    void * handle,
    const char * text,
    float * out_vec,
    int max_dim
) {
    if (!handle || !text || !out_vec) {
        return -1;
    }
    auto * ctx = (docneedle_embed_ctx *)handle;
    std::lock_guard<std::mutex> lock(ctx->mtx);

    const llama_vocab * vocab = llama_model_get_vocab(ctx->model);
    std::string prompt(text);
    std::vector<llama_token> tokens(prompt.length() + 16);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), true, true);
    } else {
        tokens.resize(n_tokens);
    }
    if (n_tokens <= 0) {
        return -2;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n_tokens;

    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    int dec_res = llama_decode(ctx->lctx, batch);
    llama_batch_free(batch);
    if (dec_res != 0) {
        return -3;
    }

    const float * emb = llama_get_embeddings_seq(ctx->lctx, 0);
    if (!emb) {
        emb = llama_get_embeddings_ith(ctx->lctx, -1);
    }
    if (!emb) {
        emb = llama_get_embeddings(ctx->lctx);
    }
    if (!emb) {
        return -4;
    }

    int dim = (max_dim < ctx->dimension) ? max_dim : ctx->dimension;
    std::memcpy(out_vec, emb, dim * sizeof(float));
    normalize_l2(out_vec, dim);
    return 0;
}

int docneedle_embed_image(
    void * handle,
    const unsigned char * img_data,
    size_t img_len,
    float * out_vec,
    int max_dim
) {
    if (!handle || !img_data || img_len == 0 || !out_vec) {
        return -1;
    }
    auto * ctx = (docneedle_embed_ctx *)handle;
    std::lock_guard<std::mutex> lock(ctx->mtx);

    struct mtmd_helper_init_opt opt = mtmd_helper_init_opt_default();
    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(
        ctx->mctx,
        img_data,
        img_len,
        false,
        opt
    );
    if (!bw.bitmap) {
        return -2;
    }

    const char * marker = mtmd_get_marker(ctx->mctx);
    std::string prompt(marker ? marker : "<|image>");
    mtmd_input_text input_text = { prompt.c_str(), prompt.length(), true, true };
    const mtmd_bitmap * bitmaps[] = { bw.bitmap };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int tok_res = mtmd_tokenize(ctx->mctx, chunks, &input_text, bitmaps, 1);
    if (tok_res != 0) {
        mtmd_bitmap_free(bw.bitmap);
        mtmd_input_chunks_free(chunks);
        return -3;
    }

    llama_memory_clear(llama_get_memory(ctx->lctx), true);
    llama_pos new_past = 0;
    int eval_res = mtmd_helper_eval_chunks(ctx->mctx, ctx->lctx, chunks, 0, 0, 2048, true, &new_past);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_input_chunks_free(chunks);
    if (eval_res != 0) {
        return -4;
    }

    const float * emb = llama_get_embeddings_seq(ctx->lctx, 0);
    if (!emb) {
        emb = llama_get_embeddings_ith(ctx->lctx, -1);
    }
    if (!emb) {
        emb = llama_get_embeddings(ctx->lctx);
    }
    if (!emb) {
        return -5;
    }

    int dim = (max_dim < ctx->dimension) ? max_dim : ctx->dimension;
    std::memcpy(out_vec, emb, dim * sizeof(float));
    normalize_l2(out_vec, dim);
    return 0;
}

void docneedle_embed_free(void * handle) {
    if (!handle) {
        return;
    }
    auto * ctx = (docneedle_embed_ctx *)handle;
    if (ctx->mctx) {
        mtmd_free(ctx->mctx);
    }
    if (ctx->lctx) {
        llama_free(ctx->lctx);
    }
    if (ctx->model) {
        llama_model_free(ctx->model);
    }
    delete ctx;
}

}
