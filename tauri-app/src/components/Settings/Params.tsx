import { defaultParams } from "../../config";
import { parameterProps, useStore, Params as ParamsType } from "../../hooks/useStore";
import { Section } from "./Section";

export const Params = () => {
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
          <div key={id} className=" w-full space-y-1">
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
