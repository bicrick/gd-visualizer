import React from 'react';
import { ControlGroup } from '../UI/ControlGroup';
import { OptimizerSection } from './OptimizerSection';
import { OPTIMIZER_CONFIGS } from './optimizerConfigs';

export const Optimizers: React.FC = () => {
  return (
    <ControlGroup title="Optimizers">
      {OPTIMIZER_CONFIGS.map((config) => (
        <OptimizerSection key={config.id} config={config} />
      ))}
    </ControlGroup>
  );
};
