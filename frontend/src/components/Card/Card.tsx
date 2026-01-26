import { useState, ReactNode } from 'react'
import styles from './Card.module.css'

interface CardProps {
  title: string
  summary?: ReactNode
  children: ReactNode
  defaultCollapsed?: boolean
  className?: string
  collapsedControls?: ReactNode
}

export function Card({ 
  title, 
  summary, 
  children, 
  defaultCollapsed = false,
  className = '',
  collapsedControls
}: CardProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed)
  
  return (
    <div className={`${styles.card} ${className}`}>
      <div 
        className={styles.header}
      >
        <div 
          className={styles.headerContent}
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <span className={styles.title}>{title}</span>
          {isCollapsed && summary && (
            <span className={styles.summary}>{summary}</span>
          )}
        </div>
        {isCollapsed && collapsedControls && (
          <div className={styles.collapsedControls} onClick={(e) => e.stopPropagation()}>
            {collapsedControls}
          </div>
        )}
        <span 
          className={`${styles.arrow} ${isCollapsed ? styles.collapsed : ''}`}
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          ▼
        </span>
      </div>
      
      <div className={`${styles.content} ${isCollapsed ? styles.contentCollapsed : ''}`}>
        <div className={styles.contentInner}>
          {children}
        </div>
      </div>
    </div>
  )
}
