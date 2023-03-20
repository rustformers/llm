import { invoke } from "@tauri-apps/api";
import { listen } from "@tauri-apps/api/event";
import { useEffect } from "react";
import { defaultParams } from "../config";
import { InputParams, useStore } from "./useStore";
import { toast } from "sonner";

export const complete = async (params: InputParams) => {
  return await invoke("complete", { params: { ...defaultParams, ...params } });
};
type Payload = {
  message: string;
  id: string;
};
type Error = {
  message: string;
};
export const useListener = () => {
  const editMessage = useStore((s) => s.editMessage);

  useEffect(() => {
    listen<Payload>("message", (event) => {
      editMessage(event.payload.id, event.payload.message);
    });
    listen<Error>("toast", (event) => {
      toast.error(event.payload.message);
    });
  }, []);
};
