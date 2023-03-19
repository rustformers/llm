import { create } from "zustand";
import { persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { open } from "@tauri-apps/api/dialog";
import { getRandomId } from "../helpers";
import { defaultParams, defaultPrompt } from "../config";

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
export type InputParams = Params & {
  path: string;
  prompt: string;
  id: string;
};

export const parameterProps: { [id in keyof Params]: { label: string; placeholder?: string } } = {
  n_batch: { label: "Batch size" },
  n_threads: { label: "Threads", placeholder: "Max" },
  top_k: { label: "Top K" },
  top_p: { label: "Top P" },
  repeat_penalty: { label: "Repeat penalty" },
  temp: { label: "Temperature" },
  num_predict: { label: "Number of predictions" },
};

export type Message = {
  id: string;
  message: string;
  type: "user" | "asssistant";
  isGenerating?: boolean;
};

export type Store = {
  params: Params;
  setParams: (params: Partial<Params>) => void;
  prompt: string;
  setPrompt: (prompt: string) => void;
  resetPrompt: () => void;

  selectedModel?: string;
  setSelectedModel: (id: string) => void;

  models: { [id: string]: Model };
  editModel: (model: Model) => void;
  removeModel: (id: string) => void;
  addModel: () => void;

  messages: { [id: string]: Message };
  allMessages: string[];
  addMessage: (message: Message) => void;
  editMessage: (id: string, message: string) => void;
  removeMessage: (id: string) => void;
  clearMessages: () => void;
};

export const useStore = create(
  persist(
    immer<Store>((set) => ({
      params: defaultParams,
      setParams: (params) => set((state) => ({ params: { ...state.params, ...params } })),

      prompt: defaultPrompt,
      setPrompt: (prompt) => set({ prompt }),
      resetPrompt: () => set({ prompt: defaultPrompt }),

      setSelectedModel: (modelPath) => set({ selectedModel: modelPath }),
      models: {},
      editModel: (model) =>
        set((state) => {
          state.models[model.id] = model;
        }),
      removeModel: (id) =>
        set((state) => {
          delete state.models[id];
          state.selectedModel = Object.keys(state.models)[0] || undefined;
        }),
      addModel: async () => {
        const path = await open({
          directory: false,
          multiple: false,
          title: "Select Model",
          filters: [{ name: "Model", extensions: ["bin"] }],
        });
        if (!path || Array.isArray(path)) return;
        const name = path.split("/").pop()?.split(".")[0] || path;
        const id = getRandomId();
        set((state) => {
          state.models[id] = { name, path, id };
          state.selectedModel = id;
        });
      },
      clearMessages: () => set({ messages: {}, allMessages: [] }),

      messages: {},
      allMessages: [],
      addMessage: (message) =>
        set((state) => {
          state.messages[message.id] = message;
          state.allMessages.push(message.id);
        }),
      editMessage: (id, message) =>
        set((state) => {
          state.messages[id] = { ...state.messages[id], message };
        }),
      removeMessage: (id) =>
        set((state) => {
          delete state.messages[id];
          state.allMessages = state.allMessages.filter((m) => m !== id);
        }),
    })),
    { name: "llama-rs" }
  )
);

export const useMessage = (id: string): Message | undefined => useStore((s) => s.messages[id]);
export const useModel = (id?: string): Model | undefined => useStore((s) => s.models[id || s.selectedModel || ""]);
