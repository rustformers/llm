import { Main } from "./components/Main";
import { Menu } from "./components/Menu";

export default function App() {
  return (
    <div className="grid grid-cols-3 h-screen overflow-hidden">
      <Menu />
      <Main />
    </div>
  );
}
