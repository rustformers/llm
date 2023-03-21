import { Params } from "./types";

export const alpacaName = "ggml-alpaca-7b-q4.bin";
export const alpacaUrl = "https://gateway.estuary.tech/gw/ipfs/QmQ1bf2BTnYxq73MFJWu1B7bQ2UD6qG7D7YDCxhTndVkPC";

export const defaultPrompt = {
  instruction: "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
  userPrefix: "### Instruction:",
  assistantPrefix: "### Response:",
};

export const defaultParams: Params = {
  n_batch: 8,
  top_k: 40,
  top_p: 0.95,
  repeat_penalty: 1.3,
  temp: 0.8,
  num_predict: 512,
};

export const parameterDetails: { [id in keyof Params]: { label: string; placeholder?: string } } = {
  n_batch: { label: "Batch size" },
  n_threads: { label: "Threads", placeholder: "Max" },
  top_k: { label: "Top K" },
  top_p: { label: "Top P" },
  repeat_penalty: { label: "Repeat penalty" },
  temp: { label: "Temperature" },
  num_predict: { label: "Number of predictions" },
};
