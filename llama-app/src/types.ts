export type Model = {
  name: string;
  path: string;
  id: string;
};

export type Params = {
  n_batch?: number;
  n_threads?: number;
  top_k?: number;
  top_p?: number;
  repeat_penalty?: number;
  temp?: number;
  num_predict?: number;
};
export type Prompt = {
  instruction: string;
  userPrefix: string;
  assistantPrefix: string;
};
export type InputParams = Params & {
  path: string;
  prompt: string;
  id: string;
};

export type MessageType = "user" | "asssistant";
export type Message = {
  id: string;
  message: string;
  type: MessageType;
  index: number;
};
