import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
// @ts-ignore - OrbitControls doesn't have types in the examples
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { useAppContext } from '../context/AppContext';
import { api } from '../services/api';
import { LandscapeData } from '../context/types';

export const useThreeScene = (canvasRef: React.RefObject<HTMLCanvasElement | null>) => {
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<any>(null);
  const raycasterRef = useRef<THREE.Raycaster | null>(null);
  const landscapeMeshRef = useRef<THREE.Mesh | null>(null);
  const startMarkerRef = useRef<THREE.Mesh | null>(null);
  const gridHelperRef = useRef<THREE.GridHelper | null>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [landscapeZRange, setLandscapeZRange] = useState({ zMin: 0, zMax: 1, scale: 2.0 });

  const {
    theme,
    currentManifoldId,
    manifoldParams,
    setStartPosition,
    pickingMode,
    setPickingMode,
  } = useAppContext();

  // Initialize Three.js scene
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    if (!container) return;

    // Scene
    const scene = new THREE.Scene();
    const bgColor = theme === 'dark' ? 0x0a0a0a : 0xfafafa;
    scene.background = new THREE.Color(bgColor);
    sceneRef.current = scene;

    // Camera
    const width = container.clientWidth;
    const height = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(15, 15, 15);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 50;
    controlsRef.current = controls;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(10, 10, 5);
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-10, -10, -5);
    scene.add(directionalLight2);

    // Grid helper
    const gridColor1 = theme === 'dark' ? 0x444444 : 0xcccccc;
    const gridColor2 = theme === 'dark' ? 0x222222 : 0xe8e8e8;
    const gridHelper = new THREE.GridHelper(20, 20, gridColor1, gridColor2);
    scene.add(gridHelper);
    gridHelperRef.current = gridHelper;

    // Axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // Raycaster for point picking
    const raycaster = new THREE.Raycaster();
    raycasterRef.current = raycaster;

    // Handle window resize
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', handleResize);

    // Animation loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
      controls.dispose();
    };
  }, [canvasRef, theme]);

  // Update theme colors
  useEffect(() => {
    if (!sceneRef.current || !gridHelperRef.current) return;

    const bgColor = theme === 'dark' ? 0x0a0a0a : 0xfafafa;
    sceneRef.current.background = new THREE.Color(bgColor);

    // Update grid
    const gridColor1 = theme === 'dark' ? 0x444444 : 0xcccccc;
    const gridColor2 = theme === 'dark' ? 0x222222 : 0xe8e8e8;
    sceneRef.current.remove(gridHelperRef.current);
    const newGrid = new THREE.GridHelper(20, 20, gridColor1, gridColor2);
    sceneRef.current.add(newGrid);
    gridHelperRef.current = newGrid;
  }, [theme]);

  // Load landscape when manifold changes
  useEffect(() => {
    if (!currentManifoldId || !sceneRef.current) return;

    const loadLandscape = async () => {
      setIsLoading(true);
      try {
        const data = await api.getLandscape(currentManifoldId, manifoldParams);
        
        // Calculate zMin and zMax if not provided by backend
        if (!data.zMin || !data.zMax) {
          let min = Infinity;
          let max = -Infinity;
          for (let i = 0; i < data.z.length; i++) {
            for (let j = 0; j < data.z[i].length; j++) {
              const val = data.z[i][j];
              if (val < min) min = val;
              if (val > max) max = val;
            }
          }
          data.zMin = min;
          data.zMax = max;
        }
        
        createLandscapeMesh(data);
      } catch (error) {
        console.error('Error loading landscape:', error);
      } finally {
        setIsLoading(false);
      }
    };

    loadLandscape();
  }, [currentManifoldId, manifoldParams]);

  // Create landscape mesh from data
  const createLandscapeMesh = (data: LandscapeData) => {
    if (!sceneRef.current) return;

    // Remove old mesh
    if (landscapeMeshRef.current) {
      sceneRef.current.remove(landscapeMeshRef.current);
      landscapeMeshRef.current.geometry.dispose();
      if (Array.isArray(landscapeMeshRef.current.material)) {
        landscapeMeshRef.current.material.forEach((m) => m.dispose());
      } else {
        landscapeMeshRef.current.material.dispose();
      }
    }

    const { x, y, z, zMin, zMax } = data;
    const width = x.length;
    const height = y.length;

    setLandscapeZRange({ zMin, zMax, scale: 2.0 });

    // Create geometry
    const geometry = new THREE.PlaneGeometry(10, 10, width - 1, height - 1);
    const positions = geometry.attributes.position;

    // Apply z values and colors
    const colors = [];
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const idx = i * width + j;
        const zValue = (z[i][j] - zMin) / (zMax - zMin) * 2.0;
        positions.setZ(idx, zValue);

        // Color gradient from blue (low) to red (high)
        const normalized = (z[i][j] - zMin) / (zMax - zMin);
        const color = new THREE.Color();
        color.setHSL(0.6 - normalized * 0.6, 1, 0.5);
        colors.push(color.r, color.g, color.b);
      }
    }

    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshPhongMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      flatShading: false,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    sceneRef.current.add(mesh);
    landscapeMeshRef.current = mesh;
  };

  // Update start point marker
  const updateStartPointMarker = (x: number, y: number) => {
    if (!sceneRef.current) return;

    // Remove old marker
    if (startMarkerRef.current) {
      sceneRef.current.remove(startMarkerRef.current);
    }

    // Get height at position
    const worldCoords = paramsToWorldCoords(x, y, 0);

    // Create marker
    const geometry = new THREE.ConeGeometry(0.2, 0.6, 8);
    const material = new THREE.MeshPhongMaterial({
      color: 0xffff00,
      emissive: 0xffff00,
      emissiveIntensity: 0.5,
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.position.set(worldCoords.x, worldCoords.y + 1, worldCoords.z);
    marker.rotation.x = Math.PI;
    sceneRef.current.add(marker);
    startMarkerRef.current = marker;
  };

  // Convert parameter space to world coordinates
  const paramsToWorldCoords = (x: number, y: number, loss: number) => {
    const range = [-5, 5]; // Default range
    const rangeMin = range[0];
    const rangeMax = range[1];
    const rangeSize = rangeMax - rangeMin;

    const worldX = ((x - rangeMin) / rangeSize - 0.5) * 10;
    const worldZ = ((y - rangeMin) / rangeSize - 0.5) * 10;

    let worldY = 0;
    if (landscapeMeshRef.current) {
      // Sample height from landscape
      worldY = (loss - landscapeZRange.zMin) / (landscapeZRange.zMax - landscapeZRange.zMin) * landscapeZRange.scale;
    }

    return { x: worldX, y: worldY, z: worldZ };
  };

  // Handle canvas click for point picking
  useEffect(() => {
    if (!canvasRef.current || !raycasterRef.current) return;

    const handleClick = (event: MouseEvent) => {
      if (!pickingMode || !landscapeMeshRef.current || !cameraRef.current) return;

      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );

      raycasterRef.current!.setFromCamera(mouse, cameraRef.current);
      const intersects = raycasterRef.current!.intersectObject(landscapeMeshRef.current);

      if (intersects.length > 0) {
        const point = intersects[0].point;
        // Convert world coords back to parameter space
        const x = (point.x / 10 + 0.5) * 10 - 5;
        const y = (point.z / 10 + 0.5) * 10 - 5;
        setStartPosition(x, y);
        updateStartPointMarker(x, y);
        setPickingMode(false);
      }
    };

    const canvas = canvasRef.current;
    canvas.addEventListener('click', handleClick);

    return () => {
      canvas.removeEventListener('click', handleClick);
    };
  }, [pickingMode, canvasRef, setStartPosition, setPickingMode]);

  return {
    scene: sceneRef.current,
    camera: cameraRef.current,
    renderer: rendererRef.current,
    controls: controlsRef.current,
    landscapeMesh: landscapeMeshRef.current,
    isLoading,
    landscapeZRange,
    updateStartPointMarker,
    paramsToWorldCoords,
  };
};
