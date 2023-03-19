import { invoke } from "@tauri-apps/api";
import { listen } from "@tauri-apps/api/event";
import { useEffect } from "react";
import { getRandomId } from "../helpers";
import { InputParams, useModel, useStore } from "./useStore";

export const complete = async (params: InputParams) => {
  return await invoke("complete", { params });
};
type Payload = {
  message: string;
  id: string;
};
export const useComplete = () => {
  const addMessage = useStore((s) => s.addMessage);
  const editMessage = useStore((s) => s.editMessage);
  const params = useStore((s) => s.params);
  const model = useModel();

  const send = async (prompt: string) => {
    if (!model) return alert("No model selected");
    const id = getRandomId();
    addMessage({ id, message: "", type: "asssistant" });
    const res = await complete({ prompt, id, path: model?.path, ...params });
    console.log(res);
  };
  
  useEffect(() => {
    listen<Payload>("message", (event) => {
      editMessage(event.payload.id, event.payload.message);
    });
  }, []);
  return send;
};
