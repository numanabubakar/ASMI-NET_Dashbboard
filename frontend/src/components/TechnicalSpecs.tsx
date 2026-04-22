'use client';

import React from 'react';
import { Card } from './ui/card';
import { Cpu, Layers, Network, Database } from 'lucide-react';

export function TechnicalSpecs() {
  const specs = [
    { icon: <Cpu className="w-4 h-4" />, label: "Backbone", value: "ResNet50" },
    { icon: <Layers className="w-4 h-4" />, label: "Feature Extraction", value: "Dynamic FPN" },
    { icon: <Network className="w-4 h-4" />, label: "Graph Reasoner", value: "Multi-head GAT" },
    { icon: <Database className="w-4 h-4" />, label: "Dataset", value: "MLRSNet" }
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {specs.map((spec) => (
        <Card key={spec.label} className="bg-slate-900/50 border-slate-800 p-3 flex flex-col items-center justify-center text-center space-y-1">
          <div className="p-1.5 rounded-md bg-indigo-500/10 text-indigo-400">
            {spec.icon}
          </div>
          <p className="text-[10px] uppercase tracking-wider text-slate-500 font-bold">
            {spec.label}
          </p>
          <p className="text-xs text-slate-200 font-medium">
            {spec.value}
          </p>
        </Card>
      ))}
    </div>
  );
}
