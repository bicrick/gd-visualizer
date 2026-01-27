import { useState, useRef, useEffect } from 'react'
import { useOptimizerStore, OPTIMIZER_COLORS } from '../../stores'
import styles from './OptimizerLegend.module.css'

const SNAP_ZONE_TOP = 100
const SNAP_ZONE_HORIZONTAL = 150
const DOCKED_TOP = 20

export function OptimizerLegend() {
  const enabled = useOptimizerStore(state => state.enabled)
  const [isDragging, setIsDragging] = useState(false)
  const [isDocked, setIsDocked] = useState(true)
  const [isInSnapZone, setIsInSnapZone] = useState(false)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const dragRef = useRef<HTMLDivElement>(null)
  const dragStartRef = useRef({ x: 0, y: 0, elementX: 0, elementY: 0 })
  
  // Calculate centered position for docked mode
  const getDockedPosition = () => {
    if (!dragRef.current) return { x: 0, y: DOCKED_TOP }
    const rect = dragRef.current.getBoundingClientRect()
    const centerX = (window.innerWidth - rect.width) / 2
    return { x: centerX, y: DOCKED_TOP }
  }
  
  const enabledOptimizers = Object.entries(enabled)
    .filter(([_, isEnabled]) => isEnabled)
    .map(([name]) => ({
      name,
      color: OPTIMIZER_COLORS[name] || '#ffffff',
      displayName: name.charAt(0).toUpperCase() + name.slice(1)
    }))
  
  // Trigger re-render after mount to ensure centered position is calculated with correct dimensions
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])
  
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !dragRef.current) return
      
      const deltaX = e.clientX - dragStartRef.current.x
      const deltaY = e.clientY - dragStartRef.current.y
      
      const newX = dragStartRef.current.elementX + deltaX
      const newY = dragStartRef.current.elementY + deltaY
      
      // Constrain to viewport
      const rect = dragRef.current.getBoundingClientRect()
      const maxX = window.innerWidth - rect.width
      const maxY = window.innerHeight - rect.height
      
      const constrainedX = Math.max(0, Math.min(newX, maxX))
      const constrainedY = Math.max(0, Math.min(newY, maxY))
      
      setPosition({ x: constrainedX, y: constrainedY })
      
      // Check if in snap zone for visual feedback
      const centerX = constrainedX + rect.width / 2
      const viewportCenterX = window.innerWidth / 2
      const isNearTop = constrainedY < SNAP_ZONE_TOP
      const isNearCenter = Math.abs(centerX - viewportCenterX) < SNAP_ZONE_HORIZONTAL
      
      setIsInSnapZone(isNearTop && isNearCenter)
    }
    
    const handleMouseUp = (e: MouseEvent) => {
      if (!isDragging || !dragRef.current) return
      
      setIsDragging(false)
      setIsInSnapZone(false)
      
      // Check if we should snap back to docked position
      const rect = dragRef.current.getBoundingClientRect()
      const centerX = rect.left + rect.width / 2
      const viewportCenterX = window.innerWidth / 2
      
      const isNearTop = rect.top < SNAP_ZONE_TOP
      const isNearCenter = Math.abs(centerX - viewportCenterX) < SNAP_ZONE_HORIZONTAL
      
      if (isNearTop && isNearCenter) {
        // Snap back to docked position
        setIsDocked(true)
        setPosition({ x: 0, y: 0 })
      } else {
        // Stay floating
        setIsDocked(false)
      }
    }
    
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging])
  
  const handleMouseDown = (e: React.MouseEvent) => {
    if (!dragRef.current) return
    
    // Capture current position (whether docked or floating)
    const currentPos = isDocked ? getDockedPosition() : position
    
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      elementX: currentPos.x,
      elementY: currentPos.y
    }
    
    if (isDocked) {
      setIsDocked(false)
      setPosition(currentPos)
    }
    
    setIsDragging(true)
    e.preventDefault()
  }
  
  // Handle window resize - force re-render to recalculate docked position
  useEffect(() => {
    const handleResize = () => {
      if (isDocked) {
        // Force re-render to recalculate centered position
        setPosition(prev => ({ ...prev }))
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [isDocked])
  
  if (enabledOptimizers.length === 0) {
    return null
  }
  
  const legendClasses = [
    styles.legend,
    isDragging ? styles.dragging : styles.draggable
  ].filter(Boolean).join(' ')
  
  // Always use fixed positioning, calculate position based on docked state
  const currentPosition = isDocked ? getDockedPosition() : position
  
  const legendStyle = {
    position: 'fixed' as const,
    left: `${currentPosition.x}px`,
    top: `${currentPosition.y}px`,
    zIndex: 100
  }
  
  // Ghost element styling (shown at docked position while dragging)
  const dockedPos = getDockedPosition()
  const ghostStyle = {
    position: 'fixed' as const,
    left: `${dockedPos.x}px`,
    top: `${dockedPos.y}px`,
    zIndex: 99
  }
  
  const ghostClasses = [
    styles.legend,
    styles.ghost,
    isInSnapZone && styles.snapActive
  ].filter(Boolean).join(' ')
  
  return (
    <>
      {/* Ghost outline shown at docked position while dragging */}
      {isDragging && dragRef.current && (
        <div 
          className={ghostClasses}
          style={{
            ...ghostStyle,
            width: `${dragRef.current.offsetWidth}px`,
            height: `${dragRef.current.offsetHeight}px`
          }}
        />
      )}
      
      {/* Actual draggable legend */}
      <div 
        ref={dragRef}
        className={legendClasses}
        style={legendStyle}
        onMouseDown={handleMouseDown}
      >
        {enabledOptimizers.map(({ name, color, displayName }) => (
          <div key={name} className={styles.item}>
            <div 
              className={styles.colorDot} 
              style={{ backgroundColor: color }}
            />
            <span className={styles.name}>{displayName}</span>
          </div>
        ))}
      </div>
    </>
  )
}
