import React, { useCallback } from 'react';
import { InfoIcon } from './InfoIcon';

interface ParameterSliderProps {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  tooltip?: string;
  formatValue?: (value: number) => string;
}

export const ParameterSlider: React.FC<ParameterSliderProps> = ({
  id,
  label,
  value,
  min,
  max,
  step,
  onChange,
  tooltip,
  formatValue,
}) => {
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(parseFloat(e.target.value));
    },
    [onChange]
  );

  const displayValue = formatValue ? formatValue(value) : value.toString();

  return (
    <div className="control-item">
      <label htmlFor={id}>
        {label}:
        {tooltip && <InfoIcon tooltip={tooltip} />}
      </label>
      <input
        type="range"
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
      />
      <span id={`${id}-value`}>{displayValue}</span>
    </div>
  );
};
