# GD Visualizer Refactor Plan

## Main Goals

- [ ] **Migrate to Vite + React** - Break up the monolithic index.html into modular components

- [ ] **Mobile-First UI Refactor** - No scrolling sidebar, works great on mobile. Consider floating cards or settings cog for advanced options.

- [ ] **Remove Two-Circle Clustering** - Clean up the classifier visualization code

- [ ] **Add Real ML Problems** - Implement actual ML loss landscapes (linear regression, logistic regression, simple neural nets, etc.) with 2-parameter constraint for clean visualization

- [ ] **Keep Existing Manifolds** - Preserve Gaussian Wells, Himmelblau, Rastrigin, Ackley functions

- [ ] **Handle Multi-Parameter Problems** - Figure out visualization strategy for >2 parameters (partial derivatives? 2D slices? parameter pair selection?)
