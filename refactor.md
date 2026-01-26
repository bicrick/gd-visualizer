# GD Visualizer Refactor Plan

## Main Goals

- [ ] **Migrate to Vite + React** - Break up the monolithic index.html into modular components

- [ ] **Mobile-First UI Refactor** - No scrolling sidebar, works great on mobile. Consider floating cards or settings cog for advanced options.

- [ ] **Add Real ML Problems** - Implement actual ML loss landscapes (linear regression, logistic regression, simple neural nets, etc.) with 2-parameter constraint for clean visualization

- [ ] **Keep Existing Manifolds** - Preserve Gaussian Wells, Himmelblau, Rastrigin, Ackley functions

- [ ] **Handle Multi-Parameter Problems** - Figure out visualization strategy for >2 parameters (partial derivatives? 2D slices? parameter pair selection?)

- [ ] **Add random parameter cycle** - Make the deafult gaussian wells parameters more intereseting. Also add a randomness/cylce button that will randomize tehe parameters

- [ ] **Add parametrs to the other manifolds** - Make the other manifolds move faster.

- [ ] **Add (i) buttons to each optimizer showing in depth detail on how it works** - Add info cards to each optimizer so that when you click on them, you get in depth understandings of how each optimzier works, and how they differ. Why one might be faster/slower. The performance benefits over the others etc.

