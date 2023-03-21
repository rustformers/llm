import { create } from "zustand";
import { persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { open } from "@tauri-apps/api/dialog";
import { getPrompt, getRandomId } from "../helpers";
import { defaultParams, defaultPrompt } from "../config";
import { InputParams, Message, MessageType, Model, Params, Prompt } from "../types";
import { invoke } from "@tauri-apps/api";

export type Store = {
  params: Params;
  setParams: (params: Partial<Params>) => void;

  messageCounter: number;

  prompt: Prompt;
  setPrompt: (prompt: Partial<Prompt>) => void;
  resetPrompt: () => void;

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

  send: (message: Message) => Promise<void>;
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
          if (!isActive) s.messageCounter = 0;
        }),

      prompt: defaultPrompt,
      setPrompt: (prompt) =>
        set((s) => {
          s.prompt = { ...s.prompt, ...prompt };
        }),
      resetPrompt: () => set({ prompt: defaultPrompt }),

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
      send: async (input: Message) => {
        const model = get().models[get().selectedModel || ""];
        const { id } = get().addMessage("", "asssistant");
        const inputPrompt = getPrompt(get().prompt, input.message, input.index === 0);
        const params: InputParams = { prompt: inputPrompt, id, path: model.path, ...get().params };
        return await invoke("complete", { params: { ...defaultParams, ...params } });
      },
    })),
    { name: "llama-rs" }
  )
);

export const useMessage = (id: string): Message | undefined => useStore((s) => s.messages[id]);
export const useModel = (id?: string): Model | undefined => useStore((s) => s.models[id || s.selectedModel || ""]);
