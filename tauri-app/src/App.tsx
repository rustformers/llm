import { useEffect, useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";
import { listen } from "@tauri-apps/api/event";

function App() {
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  const input = {
    path: "/Users/karelnagel/Documents/projects/llama-rs/models/Alpaca/ggml-alpaca-7b-q4.bin",
    prompt,
    n_batch: 8,
    n_threads: 4,
    top_k: 40,
    top_p: 0.95,
    repeat_penalty: 1.3,
    temp: 0.8,
    num_predict: 512,
  };

  async function greet() {
    setAnswer("");
    invoke("complete", { input });
  }
  useEffect(() => {
    listen("message", (event) => {
      console.log(event);
      setAnswer((event.payload as any)?.message);
    });
  }, []);

  return (
    <div className="container">
      <h1>Welcome to Tauri!</h1>

      <div className="row">
        <a href="https://vitejs.dev" target="_blank">
          <img src="/vite.svg" className="logo vite" alt="Vite logo" />
        </a>
        <a href="https://tauri.app" target="_blank">
          <img src="/tauri.svg" className="logo tauri" alt="Tauri logo" />
        </a>
        <a href="https://reactjs.org" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>

      <p>Click on the Tauri, Vite, and React logos to learn more.</p>

      <div className="row">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            greet();
          }}
        >
          <input id="greet-input" onChange={(e) => setPrompt(e.currentTarget.value)} placeholder="Enter a name..." />
          <button type="submit">Greet</button>
        </form>
      </div>
      <p>{answer}</p>
    </div>
  );
}

export default App;
