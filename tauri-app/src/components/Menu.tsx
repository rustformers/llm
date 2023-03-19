import { defaultParams, parameterProps, Params as ParamsType, useStore } from "../hooks/useStore";
import { useState } from "react";
import { IoIosArrowDown, IoIosTrash, IoIosAdd } from "react-icons/io";

export const Menu = () => {
  return (
    <div className="border-r h-full p-2 space-y-2 overflow-auto">
      <Models />
      <Params />
    </div>
  );
};

const Section = ({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) => {
  const [open, setOpen] = useState(true);
  return (
    <div className="flex flex-col bg-zinc-200 rounded-lg overflow-hidden">
      <div className="flex justify-between items-center p-2 cursor-pointer bg-zinc-300" onClick={() => setOpen(!open)}>
        <h3 className="text-lg font-semibold text-primary-content">{title}</h3>
        <IoIosArrowDown className={`duration-150 ${open ? "-rotate-180" : ""}`} />
      </div>
      {open && <div className={className}>{children}</div>}
    </div>
  );
};

const Models = () => {
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
      <div onClick={addNew} className="flex justify-center p-2 hover:bg-blue-200 cursor-pointer">
        <IoIosAdd className="text-xl" />
      </div>
    </Section>
  );
};

const Params = () => {
  const params = useStore((state) => state.params);
  const setParams = useStore((state) => state.setParams);
  return (
    <Section title="Params" className="flex flex-col space-y-2 p-2">
      {Object.keys(parameterProps).map((id) => {
        const key = id as keyof ParamsType;
        const props = parameterProps[key];
        const label = props?.label;
        const value = params[key];
        const placeholder = props?.placeholder || defaultParams[key];
        return (
          <div className=" w-full space-y-1">
            <label className="text-sm">{label}</label>
            <input
              type="number"
              className="w-full  rounded-md p-2 py-1"
              placeholder={placeholder?.toString()}
              value={value === undefined ? "" : value}
              onChange={(e) => setParams({ [id]: e.target.value ? Number(e.target.value) : undefined })}
            />
          </div>
        );
      })}
    </Section>
  );
};
