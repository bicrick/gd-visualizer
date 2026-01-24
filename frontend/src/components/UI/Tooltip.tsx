import React, { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

export const Tooltip: React.FC = () => {
  const [tooltip, setTooltip] = useState<{
    text: string;
    x: number;
    y: number;
    visible: boolean;
  }>({
    text: '',
    x: 0,
    y: 0,
    visible: false,
  });

  useEffect(() => {
    let hoverTimeout: number | null = null;

    const showTooltip = (event: MouseEvent, delay: number = 0) => {
      const target = event.currentTarget as HTMLElement;
      const tooltipText = target.getAttribute('data-tooltip');

      if (!tooltipText) return;

      if (hoverTimeout) {
        clearTimeout(hoverTimeout);
      }

      hoverTimeout = setTimeout(() => {
        const rect = target.getBoundingClientRect();
        setTooltip({
          text: tooltipText,
          x: rect.left + rect.width / 2,
          y: rect.top - 10,
          visible: true,
        });
      }, delay);
    };

    const hideTooltip = () => {
      if (hoverTimeout) {
        clearTimeout(hoverTimeout);
        hoverTimeout = null;
      }
      setTooltip((prev) => ({ ...prev, visible: false }));
    };

    const handleInfoIconHover = (e: MouseEvent) => showTooltip(e, 0);
    const handleButtonHover = (e: MouseEvent) => showTooltip(e, 1000);

    // Attach listeners to info icons
    document.querySelectorAll('.info-icon').forEach((icon) => {
      icon.addEventListener('mouseenter', handleInfoIconHover as EventListener);
      icon.addEventListener('mouseleave', hideTooltip);
    });

    // Attach listeners to buttons with tooltips
    document.querySelectorAll('button[data-tooltip]').forEach((button) => {
      button.addEventListener('mouseenter', handleButtonHover as EventListener);
      button.addEventListener('mouseleave', hideTooltip);
    });

    // Update position on scroll
    const handleScroll = () => {
      if (tooltip.visible) {
        hideTooltip();
      }
    };
    window.addEventListener('scroll', handleScroll, true);

    return () => {
      if (hoverTimeout) {
        clearTimeout(hoverTimeout);
      }
      document.querySelectorAll('.info-icon').forEach((icon) => {
        icon.removeEventListener('mouseenter', handleInfoIconHover as EventListener);
        icon.removeEventListener('mouseleave', hideTooltip);
      });
      document.querySelectorAll('button[data-tooltip]').forEach((button) => {
        button.removeEventListener('mouseenter', handleButtonHover as EventListener);
        button.removeEventListener('mouseleave', hideTooltip);
      });
      window.removeEventListener('scroll', handleScroll, true);
    };
  }, [tooltip.visible]);

  if (!tooltip.visible) return null;

  return createPortal(
    <div
      className="tooltip-popup"
      style={{
        left: `${tooltip.x}px`,
        top: `${tooltip.y}px`,
        transform: 'translate(-50%, -100%)',
      }}
    >
      {tooltip.text}
    </div>,
    document.body
  );
};
