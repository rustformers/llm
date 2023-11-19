use std::{env, ops::ControlFlow, path::PathBuf, process, time::Instant};

mod common;
mod llama;
mod sampling;

// total length of the sequence including the prompt
const N_LEN: usize = 32;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 1 || args[1].starts_with('-') {
        eprintln!("usage: {} MODEL_PATH [PROMPT]", args[0]);
        process::exit(1);
    }

    let mut params = common::GptParams::default();

    if args.len() >= 2 {
        params.model = PathBuf::from(args[1].clone());
    }

    if args.len() >= 3 {
        params.prompt = args[2].clone();
    }

    if params.prompt.is_empty() {
        params.prompt = String::from("Hello my name is");
    }

    // init LLM
    llama::backend_init(params.numa);

    // initialize the model
    let model_params = llama::LlamaModelParams::default();

    // model_params.n_gpu_layers = 99; // offload all layers to the GPU

    let model =
        llama::load_model_from_file(&params.model, model_params).expect("unable to load model");

    // initialize the context
    let mut ctx_params = llama::LlamaContextParams::default();

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 2048;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = if let Some(n_threads_batch) = params.n_threads_batch {
        n_threads_batch
    } else {
        params.n_threads
    };

    let mut ctx = llama::new_context_with_model(&model, ctx_params)
        .expect("failed to create the llama_context");

    // tokenize the prompt
    let tokens_list = llama::tokenize(&ctx, &params.prompt, true, None);

    let n_ctx = llama::n_ctx(&ctx);
    let n_kv_req = tokens_list.len() + (N_LEN - tokens_list.len());

    println!(
        "\n{}: n_len = {}, n_ctx = {}, n_kv_req = {}",
        file!(),
        N_LEN,
        n_ctx,
        n_kv_req
    );

    // check KV cache size
    if n_kv_req > n_ctx {
        println!(
            "{}: error: n_kv_req > n_ctx, the required KV cache size is not big enough",
            file!()
        );
        println!(
            "{}:        either reduce n_parallel or increase n_ctx",
            file!()
        );
        process::exit(1);
    }

    // print the prompt token-by-token
    eprint!("\n");
    for id in &tokens_list {
        eprint!("{}", llama::token_to_piece(&ctx, *id));
    }

    // create a llama_batch with size 512
    let mut batch = llama::LlamaBatch::new(512, 0, 1);

    // evaluate the initial prompt
    for (i, &token) in tokens_list.iter().enumerate() {
        batch.add(token, i as llama::LlamaPos, &[0; 1], false);
    }

    // llama_decode will output logits only for the last token of the prompt
    *batch.logits.last_mut().unwrap() = true;

    if llama::decode(&mut ctx, batch.clone()).is_err() {
        println!("{}: llama_decode() failed", file!());
        process::exit(1);
    }

    // main loop
    let mut n_cur = batch.tokens.len();
    let mut n_decode = 0;

    let t_main_start = Instant::now();

    while n_cur <= N_LEN {
        if decode_token(&mut ctx, &mut batch, &mut n_cur, &mut n_decode) == ControlFlow::Break(()) {
            break;
        }
    }

    println!("\n");
    let t_main_end = Instant::now();

    println!(
        "{}: decoded {} tokens in {:.2} s, speed: {:.2} t/s",
        file!(),
        n_decode,
        t_main_end.duration_since(t_main_start).as_secs_f32(),
        n_decode as f32 / t_main_end.duration_since(t_main_start).as_secs_f32()
    );

    llama::print_timings(&mut ctx);
    eprint!("\n");

    llama::backend_free();
}

fn decode_token<'a, 'b>(
    ctx: &mut llama::LlamaContext<'a>,
    batch: &'b mut llama::LlamaBatch,
    n_cur: &mut usize,
    n_decode: &mut usize,
) -> ControlFlow<(), ()> {
    // sample the next token
    let mut candidates = context_batch_token_data_array(ctx, batch);

    let new_token_id = llama::sample_token_greedy(ctx, &mut candidates);

    // check for end of stream
    if new_token_id == llama::token_eos(ctx.model) || *n_cur == N_LEN {
        println!("\n");
        return ControlFlow::Break(());
    }

    println!("{}", llama::token_to_piece(&ctx, new_token_id));
    batch.clear();

    // prepare the next batch
    batch.add(new_token_id, *n_cur as llama::LlamaToken, &[0; 1], true);
    *n_decode += 1;
    *n_cur += 1;

    // evaluate the current batch with the transformer model
    let decode = llama::decode(ctx, batch.clone());
    if decode.is_err() {
        eprintln!("{}: failed to eval, return code {}", file!(), 1);
        process::exit(1);
    }

    ControlFlow::Continue(())
}

fn context_batch_token_data_array(
    ctx: &mut llama::LlamaContext,
    batch: &llama::LlamaBatch,
) -> llama::LlamaTokenDataArray {
    let logits = llama::get_logits_ith(ctx, batch.tokens.len() - 1);
    llama::LlamaTokenDataArray {
        data: logits
            .iter()
            .enumerate()
            .map(|(token_id, logit)| llama::LlamaTokenData {
                token_id: token_id as llama::LlamaToken,
                logit: *logit,
                probability: 0.0,
            })
            .collect(),
        sorted: false,
    }
}
