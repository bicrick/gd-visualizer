# Two-Circle Clustering Problem with Local Minima

## Overview
Replace the current single-circle dataset with a two-circle clustering problem that creates multiple local minima, demonstrating the challenges of non-convex optimization.

## Implementation

### 1. New Dataset - Two Separate Circular Clusters
**File**: `backend/loss_functions.py`

Generate dataset with:
- **Cluster A**: Green points in a circle centered at (-1.5, 0), radius ~0.6
- **Cluster B**: Green points in a circle centered at (1.5, 0), radius ~0.6  
- **Orange noise**: Scattered points everywhere else
- Total: ~200 points (40 in each cluster, 120 scattered)

### 2. Updated Loss Function
**File**: `backend/loss_functions.py`

Keep the same circle classifier logic but:
- The loss landscape will naturally have TWO minima (one at each cluster)
- Add slight asymmetry (different cluster sizes or noise) to make one minimum slightly better
- This creates the classic local minima problem!

### 3. Test the Landscape
Verify that:
- Two clear valleys exist in the loss landscape
- Starting position determines which minimum optimizers find
- Creates interesting non-convex optimization demo

No frontend changes needed - visualization will automatically show the competing circles!

## Expected Result
- **Blue valley on left** (Cluster A minimum)
- **Blue valley on right** (Cluster B minimum)  
- **Ridge/saddle between them**
- Optimizers get trapped in whichever valley they start near!


