import { IoIosAdd, IoIosTrash } from "react-icons/io";
import { useStore } from "../../hooks/useStore";
import { Section } from "./Section";

export const Models = () => {
  const setSelectedModel = useStore((state) => state.setSelectedModel);
  const removeModel = useStore((state) => state.removeModel);
  const models = useStore((state) => state.models);
  const selectedModel = useStore((state) => state.selectedModel);
  const addNew = useStore((state) => state.addModel);
  return (
    <Section title="Models">
      {Object.entries(models).map(([id, model]) => (
        <div
          key={id}
          className={`p-2 flex justify-between items-center cursor-pointer ${selectedModel === id ? "bg-blue-500 text-white" : "hover:bg-blue-200"}`}
          onClick={() => setSelectedModel(id)}
        >
          <p>{model.name}</p>
          <IoIosTrash
            onClick={(e) => {
              e.stopPropagation();
              removeModel(id);
            }}
          />
        </div>
      ))}
      <div onClick={() => addNew()} className="flex justify-center p-2 hover:bg-blue-200 cursor-pointer">
        <IoIosAdd className="text-xl" />
      </div>
    </Section>
  );
};
