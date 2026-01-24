import React, { useState, useCallback, useEffect, useRef } from 'react';
import { useAppContext } from '../../context/AppContext';

export const ManifoldSelector: React.FC = () => {
  const { manifolds, currentManifoldId, setManifold } = useAppContext();
  const [isOpen, setIsOpen] = useState(false);
  const selectorRef = useRef<HTMLDivElement>(null);

  const currentManifold = manifolds.find((m) => m.id === currentManifoldId);

  const toggleDropdown = useCallback(() => {
    setIsOpen((prev) => !prev);
  }, []);

  const handleManifoldSelect = useCallback(
    (manifoldId: string) => {
      setManifold(manifoldId);
      setIsOpen(false);
    },
    [setManifold]
  );

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (selectorRef.current && !selectorRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('click', handleClickOutside);
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, []);

  return (
    <div id="manifold-selector" ref={selectorRef}>
      <div className={`manifold-current ${isOpen ? 'active' : ''}`} onClick={toggleDropdown}>
        <div className="manifold-info">
          <div className="manifold-label">Manifold:</div>
          <div id="current-manifold-name">{currentManifold?.name || 'Loading...'}</div>
        </div>
        <div className="dropdown-arrow">▼</div>
      </div>
      <div id="manifold-dropdown-list" className={`manifold-dropdown-list ${isOpen ? '' : 'hidden'}`}>
        {manifolds.map((manifold) => (
          <div
            key={manifold.id}
            className={`manifold-option ${manifold.id === currentManifoldId ? 'selected' : ''}`}
            onClick={() => handleManifoldSelect(manifold.id)}
          >
            <div className="manifold-option-name">{manifold.name}</div>
            {manifold.description && (
              <div className="manifold-option-description">{manifold.description}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
