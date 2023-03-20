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

export const parameterProps: { [id in keyof Params]: { label: string; placeholder?: string } } = {
  n_batch: { label: "Batch size" },
  n_threads: { label: "Threads", placeholder: "Max" },
  top_k: { label: "Top K" },
  top_p: { label: "Top P" },
  repeat_penalty: { label: "Repeat penalty" },
  temp: { label: "Temperature" },
  num_predict: { label: "Number of predictions" },
};
export type MessageType = "user" | "asssistant";
export type Message = {
  id: string;
  message: string;
  type: MessageType;
  index: number;
};

export type Store = {
  params: Params;
  setParams: (params: Partial<Params>) => void;
  messageCounter: number;

  prompt: Prompt;
  setPrompt: (prompt: Partial<Prompt>) => void;
  resetPrompt: () => void;

  isGenerating: boolean;
  setIsGenerating: (isGenerating: boolean) => void;

  isActive: boolean;
  setIsActive: (isActive: boolean) => void;

  selectedModel?: string;
  setSelectedModel: (id: string) => void;

  models: { [id: string]: Model };
  editModel: (id: string, name: string) => void;
  removeModel: (id: string) => void;
  addModel: (path?: string) => Promise<void>;

  messages: { [id: string]: Message };
  allMessages: string[];
  addMessage: (message: string, type: MessageType) => Message;
  editMessage: (id: string, message: string) => void;
  removeMessage: (id: string) => void;
  clearMessages: () => void;
};

export const useStore = create(
  persist(
    immer<Store>((set, get) => ({
      messageCounter: 0,
      params: defaultParams,
      setParams: (params) => set((state) => ({ params: { ...state.params, ...params } })),

      isActive: false,
      setIsActive: (isActive) =>
        set((s) => {
          if (isActive === s.isActive) return;
          s.isActive = isActive;
          s.messageCounter = 0;
        }),

      prompt: defaultPrompt,
      setPrompt: (prompt) => set((s) => ({ ...s.prompt, prompt })),
      resetPrompt: () => set({ prompt: defaultPrompt }),

      isGenerating: false,
      setIsGenerating: (isGenerating) => set({ isGenerating }),

      setSelectedModel: (modelPath) => set({ selectedModel: modelPath }),
      models: {},
      editModel: (id, name) =>
        set((state) => {
          state.models[id] = { ...state.models[id], name };
        }),
      removeModel: (id) =>
        set((state) => {
          delete state.models[id];
          state.selectedModel = Object.keys(state.models)[0] || undefined;
        }),
      addModel: async (downloaded) => {
        const path =
          downloaded ||
          (await open({
            directory: false,
            multiple: false,
            title: "Select Model",
            filters: [{ name: "Model", extensions: ["bin"] }],
          }));
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
      addMessage: (message, type) => {
        const id = getRandomId();
        const newMessage = { message, type, index: get().messageCounter, id };
        set((state) => {
          state.messages[id] = newMessage;
          state.allMessages.push(id);
          state.messageCounter++;
        });
        return newMessage;
      },
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
