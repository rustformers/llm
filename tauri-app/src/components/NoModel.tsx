import { invoke } from "@tauri-apps/api";
import { save } from "@tauri-apps/api/dialog";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";
import { alpacaName, alpacaUrl } from "../config";
import { useStore } from "../hooks/useStore";

type Progress = {
  downloaded: number;
  total_size: number;
};

export const NoModel = () => {
  const addModel = useStore((state) => state.addModel);

  const download = async () => {
    const path = await save({
      defaultPath: alpacaName,
      filters: [{ name: "Model", extensions: ["bin"] }],
      title: "Where to Save?",
    });
    await invoke("download_model", { path, url: alpacaUrl });
    await addModel(path || undefined);
  };

  const [progress, setProgress] = useState<Progress>();
  useEffect(() => {
    listen<Progress>("progress", (event) => {
      setProgress(event.payload);
    });
  }, []);

  return (
    <div className="flex flex-col items-center justify-center col-span-2 space-y-2">
      {!progress ? (
        <>
          <button className="bg-zinc-300 p-2 rounded-lg" onClick={() => addModel()}>
            Select Model From Your Computer
          </button>
          <p>or</p>
          <button onClick={download} className="bg-zinc-300 p-2 rounded-lg">
            Download Alpaca
          </button>
        </>
      ) : (
        <>
          <p>Downloading...</p>
          <div className="flex items-center space-x-3">
            <p>{((progress.downloaded / progress.total_size) * 100).toFixed(2)}%</p>
            <progress className="rounded-full bg-blue-400" value={progress.downloaded} max={progress.total_size} />
          </div>
        </>
      )}
    </div>
  );
};
