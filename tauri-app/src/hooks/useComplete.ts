import { invoke } from "@tauri-apps/api";
import { listen } from "@tauri-apps/api/event";
import { useEffect } from "react";
import { defaultParams } from "../config";
import { getPrompt, getRandomId } from "../helpers";
import { InputParams, useModel, useStore } from "./useStore";

export const complete = async (params: InputParams) => {
  return await invoke("complete", { params: { ...defaultParams, ...params } });
};
type Payload = {
  message: string;
  id: string;
};
export const useComplete = () => {
  const addMessage = useStore((s) => s.addMessage);
  const editMessage = useStore((s) => s.editMessage);
  const params = useStore((s) => s.params);
  const prompt = useStore((s) => s.prompt);
  const isGenerating = useStore((s) => s.isGenerating);
  const setIsGenerating = useStore((s) => s.setIsGenerating);
  const model = useModel();

  const send = async (instruction: string) => {
    if (!model) return;
    if (isGenerating) return;
    const { id, index } = addMessage("", "asssistant");
    const inputPrompt = getPrompt(prompt, instruction, index === 0);
    setIsGenerating(true);
    await complete({ prompt: inputPrompt, id, path: model.path, ...params });
    setIsGenerating(false);
  };

  useEffect(() => {
    setIsGenerating(false);
    listen<Payload>("message", (event) => {
      editMessage(event.payload.id, event.payload.message);
    });
  }, []);
  return send;
};
