import {
  ManifoldInfo,
  LandscapeData,
  ClassifierDataset,
  ManifoldParams,
} from '../context/types';

// In development, use relative URL to leverage Vite proxy
// In production, use environment variable or default to Cloud Run
const API_BASE_URL =
  (import.meta as any).env?.VITE_API_URL ||
  (import.meta as any).env?.MODE === 'production'
    ? 'https://gd-experiments-1031734458893.us-central1.run.app/api'
    : '/api';

export const api = {
  async getManifolds(): Promise<{ manifolds: ManifoldInfo[] }> {
    const response = await fetch(`${API_BASE_URL}/manifolds`);
    if (!response.ok) throw new Error('Failed to fetch manifolds');
    return response.json();
  },

  async getLandscape(
    manifoldId: string,
    params: ManifoldParams = {},
    resolution: number = 80
  ): Promise<LandscapeData> {
    let url = `${API_BASE_URL}/landscape?resolution=${resolution}&manifold=${encodeURIComponent(manifoldId)}`;

    if (Object.keys(params).length > 0) {
      url += `&params=${encodeURIComponent(JSON.stringify(params))}`;
    }

    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch landscape');
    return response.json();
  },

  async getClassifierDataset(): Promise<ClassifierDataset> {
    const response = await fetch(`${API_BASE_URL}/classifier_dataset`);
    if (!response.ok) throw new Error('Failed to fetch classifier dataset');
    return response.json();
  },
};
