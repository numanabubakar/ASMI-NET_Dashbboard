'use client';

import React, { useState, useEffect } from 'react';
import { Loader2 } from 'lucide-react';

export function NeuralNetworkLoader() {
  const [step, setStep] = useState(0);
  const steps = [
    "Initializing AMSI-Net architecture...",
    "Loading weights from best_model.pth...",
    "Running Dynamic Scale-Aware FPN...",
    "Computing Graph Attention Layer...",
    "Fusing Spectral & Spatial features...",
    "Generating Explanations (LIME/GradCAM)...",
    "Finalizing results..."
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev + 1) % steps.length);
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-6">
      <div className="relative">
        <Loader2 className="w-12 h-12 text-indigo-500 animate-spin" />
        <div className="absolute inset-0 blur-xl bg-indigo-500/20 rounded-full animate-pulse" />
      </div>
      <div className="text-center space-y-2">
        <h4 className="text-white font-semibold">Running Inference</h4>
        <p className="text-sm text-slate-400 font-mono italic animate-pulse">
           {steps[step]}
        </p>
      </div>
      <div className="w-full max-w-xs h-1 bg-slate-800 rounded-full overflow-hidden">
        <div 
          className="h-full bg-indigo-500 transition-all duration-500 ease-out"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }}
        />
      </div>
    </div>
  );
}
