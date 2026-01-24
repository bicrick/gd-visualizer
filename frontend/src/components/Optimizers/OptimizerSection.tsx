import React, { useState, useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';
import { OptimizerConfig, OptimizerName } from '../../context/types';
import { ParameterSlider } from '../UI/ParameterSlider';

interface OptimizerSectionProps {
  config: OptimizerConfig;
}

export const OptimizerSection: React.FC<OptimizerSectionProps> = ({ config }) => {
  const { optimizerParams, enabledOptimizers, updateOptimizerParam, toggleOptimizer } =
    useAppContext();
  const [isExpanded, setIsExpanded] = useState(false);

  const params = optimizerParams[config.id as OptimizerName];
  const isEnabled = enabledOptimizers[config.id as OptimizerName];

  const handleToggle = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      toggleOptimizer(config.id as OptimizerName, e.target.checked);
    },
    [config.id, toggleOptimizer]
  );

  const handleExpandCollapse = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const handleParamChange = useCallback(
    (paramId: string, value: number | boolean) => {
      updateOptimizerParam(config.id as OptimizerName, paramId as any, value);
    },
    [config.id, updateOptimizerParam]
  );

  const formatValue = useCallback((value: number, step: number) => {
    if (step >= 0.001 && step < 0.01) {
      return value.toFixed(3);
    } else if (step >= 0.0001 && step < 0.001) {
      return value.toFixed(4);
    } else if (step < 0.0001) {
      return value.toExponential(0);
    } else if (step < 1) {
      return value.toFixed(2);
    }
    return value.toString();
  }, []);

  return (
    <div className="optimizer-section">
      <div className="optimizer-header">
        <label className="optimizer-toggle-label">
          <input
            type="checkbox"
            id={`toggle-${config.id}`}
            className="optimizer-toggle"
            checked={isEnabled}
            onChange={handleToggle}
          />
          <span className="color-indicator" style={{ backgroundColor: config.color }}></span>
          <span className="optimizer-name">{config.name}</span>
        </label>
        <button
          className={`expand-btn ${isExpanded ? '' : 'collapsed'}`}
          data-target={`${config.id}-params`}
          onClick={handleExpandCollapse}
        >
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>
      <div className={`optimizer-params ${isExpanded ? '' : 'collapsed'}`} id={`${config.id}-params`}>
        {config.params.map((param) => {
          // Check conditional display
          if (param.conditionalDisplay) {
            const dependsOnValue = (params as any)[param.conditionalDisplay.dependsOn];
            if (dependsOnValue !== param.conditionalDisplay.value) {
              return null;
            }
          }

          if (param.type === 'checkbox') {
            return (
              <div className="control-item" key={param.id}>
                <label>
                  <input
                    type="checkbox"
                    id={`${config.id}-${param.id}`}
                    checked={(params as any)[param.id] as boolean}
                    onChange={(e) => handleParamChange(param.id, e.target.checked)}
                  />
                  {param.label}
                </label>
              </div>
            );
          }

          return (
            <ParameterSlider
              key={param.id}
              id={`${config.id}-${param.id}`}
              label={param.label}
              value={(params as any)[param.id] as number}
              min={param.min!}
              max={param.max!}
              step={param.step!}
              onChange={(value) => handleParamChange(param.id, value)}
              tooltip={param.tooltip}
              formatValue={(value) => formatValue(value, param.step!)}
            />
          );
        })}
      </div>
    </div>
  );
};
