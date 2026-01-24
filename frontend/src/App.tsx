import React, { useEffect } from 'react';
import { useAppContext } from './context/AppContext';
import { api } from './services/api';
import { Sidebar } from './components/Sidebar/Sidebar';
import { Optimizers } from './components/Optimizers/Optimizers';
import { CanvasContainer } from './components/Canvas/CanvasContainer';
import { Tooltip } from './components/UI/Tooltip';

const App: React.FC = () => {
  const { setManifolds, setManifold } = useAppContext();

  // Load manifolds on mount
  useEffect(() => {
    const loadManifolds = async () => {
      try {
        const data = await api.getManifolds();
        setManifolds(data.manifolds);
        
        // Set initial manifold
        if (data.manifolds.length > 0) {
          setManifold(data.manifolds[0].id);
        }
      } catch (error) {
        console.error('Error loading manifolds:', error);
      }
    };

    loadManifolds();
  }, [setManifolds, setManifold]);

  return (
    <div id="container">
      <Sidebar>
        <Optimizers />
      </Sidebar>
      <CanvasContainer />
      <Tooltip />
    </div>
  );
};

export default App;
