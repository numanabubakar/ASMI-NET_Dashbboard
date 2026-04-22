'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { ChevronRight, BarChart3, Eye, Info, AlertTriangle, Layers, FileJson } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ClassPrediction {
  class_index: number;
  class_label: string;
  confidence: number;
}

interface PredictionResult {
  predicted_labels: string[];
  all_predictions: ClassPrediction[];
  explainability_maps: Record<string, string>;
  uncertainty: number;
  inference_time_ms: number;
  image_info: {
    width: number;
    height: number;
    format: string;
  };
}

export function PredictionResults({ result }: { result: PredictionResult }) {
  const [activeMap, setActiveMap] = useState<string>('GradCAM');

  const maps = [...Object.keys(result.explainability_maps || []), 'Metadata'];
  
  return (
    <div className="space-y-6">
      {/* Overview Section */}
      <Card className="overflow-hidden bg-slate-900 border-slate-800">
        <CardHeader className="border-b border-slate-800 bg-slate-900/50">
          <CardTitle className="text-xl font-bold flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-400" />
            Classification Overview
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="space-y-8">
            <div className="flex flex-col md:flex-row gap-8">
              <div className="flex-1 space-y-6">
                <div>
                  <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3 block">
                    Detected Labels
                  </label>
                  <div className="flex flex-wrap gap-2">
                    {result.predicted_labels.map((label) => (
                      <span key={label} className="px-4 py-2 bg-indigo-600/20 text-indigo-400 border border-indigo-500/30 rounded-lg font-bold text-lg">
                        {label}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 hover:border-indigo-500/30 transition-colors">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                      <AlertTriangle className="w-4 h-4 text-amber-500" />
                      <span className="text-xs">Uncertainty</span>
                    </div>
                    <p className="text-xl font-mono font-bold text-amber-400">
                      {result.uncertainty.toFixed(4)}
                    </p>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-800/50 border border-slate-700 hover:border-indigo-500/30 transition-colors">
                    <div className="flex items-center gap-2 text-slate-400 mb-1">
                      <ChevronRight className="w-4 h-4 text-emerald-500" />
                      <span className="text-xs">Latency</span>
                    </div>
                    <p className="text-xl font-mono font-bold text-emerald-400">
                      {result.inference_time_ms.toFixed(1)}ms
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex-1">
                <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-4 block">
                  Top Confidence Scores
                </label>
                <div className="space-y-4">
                  {result.all_predictions.slice(0, 4).map((pred) => (
                    <div key={pred.class_label} className="space-y-1.5">
                      <div className="flex justify-between text-sm">
                        <span className={cn(
                          "font-medium",
                          result.predicted_labels.includes(pred.class_label) ? "text-indigo-400" : "text-slate-300"
                        )}>
                          {pred.class_label}
                        </span>
                        <span className="text-slate-500 font-mono">{(pred.confidence * 100).toFixed(2)}%</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700/50">
                        <div
                          className={cn(
                            "h-full transition-all duration-500 ease-out",
                            result.predicted_labels.includes(pred.class_label) ? "bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]" : "bg-slate-700"
                          )}
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Decision Intelligence Section */}
      <Card className="overflow-hidden bg-slate-900 border-slate-800">
        <CardHeader className="border-b border-slate-800 bg-slate-900/50">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <CardTitle className="text-xl font-bold flex items-center gap-2">
                <Eye className="w-5 h-5 text-indigo-400" />
                Inference Intelligence
              </CardTitle>
              <p className="text-xs text-slate-500 mt-1">
                Select a diagnostic mode to analyze model behavior and input metadata.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {maps.map(mapName => (
                <button
                  key={mapName}
                  onClick={() => setActiveMap(mapName)}
                  className={cn(
                    "px-3 py-2 rounded-xl text-xs font-bold transition-all border flex items-center gap-2",
                    activeMap === mapName 
                      ? "bg-indigo-600 text-white border-indigo-500 shadow-lg shadow-indigo-600/20" 
                      : "bg-slate-800 text-slate-400 border-slate-700 hover:bg-slate-700 hover:text-slate-200"
                  )}
                >
                  {mapName === 'Metadata' ? <FileJson className="w-3 h-3" /> : <Layers className="w-3 h-3" />}
                  {mapName}
                </button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6">
          <div className="relative aspect-square max-w-2xl mx-auto rounded-3xl border border-slate-800 overflow-hidden bg-slate-950 group shadow-2xl transition-all duration-500">
            {activeMap === 'Metadata' ? (
              <div className="w-full h-full flex flex-col items-center justify-center p-8 space-y-8 animate-in fade-in zoom-in duration-300">
                <div className="w-20 h-20 rounded-full bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                  <Info className="w-10 h-10 text-indigo-400" />
                </div>
                <div className="w-full max-w-md space-y-4">
                  <h3 className="text-xl font-bold text-white text-center mb-6">Input Image Analysis</h3>
                  {[
                    { label: "Original Resolution", value: `${result.image_info.width} × ${result.image_info.height} px` },
                    { label: "File Format", value: result.image_info.format.toUpperCase() },
                    { label: "Color Space", value: "RGB (3 Channels)" },
                    { label: "Feature Extraction", value: "Adaptive Multi-Source" },
                    { label: "Processing Mode", value: "Deep Spectral-Spatial Fusion" },
                    { label: "Engine Version", value: "AMSI-Net v1.0.4 r1" }
                  ].map((item, i) => (
                    <div key={i} className="flex justify-between items-center py-3 border-b border-slate-800/50 last:border-0 hover:bg-white/5 transition-colors px-2 rounded-lg">
                      <span className="text-sm text-slate-400">{item.label}</span>
                      <span className="text-sm text-indigo-400 font-mono font-bold">{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <>
                {result.explainability_maps[activeMap] ? (
                  <img
                    src={result.explainability_maps[activeMap]}
                    alt={activeMap}
                    className="w-full h-full object-contain transition-transform duration-700 group-hover:scale-105"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-slate-600 italic">
                    No {activeMap} data available
                  </div>
                )}
                
                <div className="absolute bottom-6 left-6 right-6 p-5 bg-slate-950/90 backdrop-blur-xl rounded-2xl border border-slate-800 transform transition-all group-hover:translate-y-[-4px] shadow-2xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-bold text-white uppercase tracking-wider">{activeMap} Visualization</span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed">
                    {activeMap === 'GradCAM' && "Dynamic gradient-weighted Class Activation Mapping: Identifies high-level visual features driving the multi-label score."}
                    {activeMap === 'GradCAM++' && "Advanced localization technique using guided backprop to handle overlapping objects in cluttered satellite imagery."}
                    {activeMap === 'LIME' && "Model-agnostic superpixel segmentation. Pinpoints which specific terrain regions correspond strictly to the detected labels."}
                    {activeMap === 'Saliency' && "Full-spectrum sensitivity analysis. Shows every pixel that influenced the neural network's decision process."}
                  </p>
                </div>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
