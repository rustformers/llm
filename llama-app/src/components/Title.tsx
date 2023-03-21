import { invoke } from "@tauri-apps/api";
import { useEffect, useState } from "react";
import { useModel, useStore } from "../hooks/useStore";

export const Title = () => {
  const model = useModel();
  const editModel = useStore((state) => state.editModel);
  const clear = useStore((state) => state.clearMessages);
  const isActive = useStore((state) => state.isActive);
  const setIsActive = useStore((state) => state.setIsActive);
  const [loading, setLoading] = useState(false);

  const toggle = async () => {
    setLoading(true);
    if (!isActive) await invoke("start_model", { path: model?.path });
    else await invoke("stop_model");

    setLoading(false);
  };

  useEffect(() => {
    const interval = setInterval(() => invoke("is_active").then((a: any) => setIsActive(a)), 500);
    return () => clearInterval(interval);
  }, []);

  if (!model) return null;
  return (
    <div className="flex items-center justify-between p-2 border-b shadow-sm">
      <h1 className="text-lg font-medium">
        <input type="text" value={model.name} onChange={(e) => editModel(model.id, e.target.value)} />
      </h1>
      <div className="flex space-x-2 text-sm">
        <button onClick={clear}>Clear</button>
        <button className="cursor-pointer" disabled={loading} onClick={toggle}>
          {isActive ? "Stop" : "Start"}
        </button>
      </div>
    </div>
  );
};