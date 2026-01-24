import React, { useState, useCallback } from 'react';
import { useAppContext } from '../../context/AppContext';
import { ParameterSlider } from '../UI/ParameterSlider';

export const ManifoldParamsPanel: React.FC = () => {
  const { manifolds, currentManifoldId, manifoldParams, updateManifoldParam } = useAppContext();
  const [isExpanded, setIsExpanded] = useState(false);

  const currentManifold = manifolds.find((m) => m.id === currentManifoldId);
  const hasParams = currentManifold?.params && currentManifold.params.length > 0;

  const toggleExpanded = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  const handleParamChange = useCallback(
    (paramId: string, value: number) => {
      updateManifoldParam(paramId, value);
    },
    [updateManifoldParam]
  );

  if (!hasParams) return null;

  return (
    <div id="manifold-params-panel" className="manifold-params-panel">
      <div className="params-header" onClick={toggleExpanded}>
        <span>Parameters</span>
        <span className="params-toggle">{isExpanded ? '▼' : '▶'}</span>
      </div>
      {isExpanded && (
        <div id="manifold-params-container">
          {currentManifold.params!.map((param) => (
            <ParameterSlider
              key={param.id}
              id={`manifold-param-${param.id}`}
              label={param.name}
              value={manifoldParams[param.id] || param.default}
              min={param.min}
              max={param.max}
              step={param.step}
              onChange={(value) => handleParamChange(param.id, value)}
            />
          ))}
        </div>
      )}
    </div>
  );
};
