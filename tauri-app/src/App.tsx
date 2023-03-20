import { Main } from "./components/Main";
import { Settings } from "./components/Settings";
import { Toaster } from "sonner";

export default function App() {
  return (
    <div className="grid grid-cols-3 h-screen overflow-hidden">
      <Settings />
      <Main />
      <Toaster position="top-center" closeButton  />
    </div>
  );
}
