'use client';

import { useState } from 'react';
import { ImageUploadZone } from '@/components/ImageUploadZone';
import { NeuralNetworkLoader } from '@/components/NeuralNetworkLoader';
import { PredictionResults } from '@/components/PredictionResults';
import { TechnicalSpecs } from '@/components/TechnicalSpecs';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Brain, Sparkles, Map, Database, LayoutDashboard, ChevronRight } from 'lucide-react';

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

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploadKey, setUploadKey] = useState(0);

  const handleImageSelected = (file: File, previewUrl: string) => {
    setSelectedFile(file);
    setPreview(previewUrl);
    setResult(null);
    setError(null);
  };

  const handleClearImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setUploadKey(prev => prev + 1);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please upload an image first');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://lulc-recognition-amsi-net.hf.space';
      const baseUrl = apiUrl.endsWith('/') ? apiUrl.slice(0, -1) : apiUrl;
      
      console.log(`[AMSI-Net] Sending request to: ${baseUrl}/predict`);

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const response = await fetch(`${baseUrl}/predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      const data = await response.json();
      console.log("[AMSI-Net] Prediction successful:", data);
      setResult(data);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : 'An error occurred during prediction';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Background Decor */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-indigo-500/10 blur-[120px] rounded-full" />
        <div className="absolute top-[20%] -right-[10%] w-[30%] h-[30%] bg-blue-500/5 blur-[100px] rounded-full" />
      </div>

      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-950/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex -space-x-2">
              <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center border border-slate-700">
                <Map className="w-4 h-4 text-indigo-400" />
              </div>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white">AMSI-Net</h1>
              <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold -mt-1">
                LULC Recognition AI
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-5xl mx-auto px-4 py-8 space-y-8 relative z-10">
        
        <div className="flex flex-col items-center text-center space-y-4 mb-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-[10px] font-bold uppercase tracking-widest">
            <Sparkles className="w-3 h-3" />
            Adaptive Multi-Source Integration
          </div>
          <h2 className="text-3xl md:text-5xl font-black text-white tracking-tight">
            Earth Observation Dashboard
          </h2>
          <p className="text-slate-400 max-w-2xl text-sm md:text-base">
            Professional multi-label classification using AMSI-Net technology.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Column 1: Control Console */}
          <div className="lg:col-span-5 space-y-6">
            <Card className="shadow-2xl shadow-indigo-500/5">
              <CardHeader>
                <CardTitle className="text-sm font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                  <LayoutDashboard className="w-4 h-4" />
                  Image Processing
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                 <ImageUploadZone
                    key={uploadKey}
                    onImageSelected={handleImageSelected}
                    onClear={handleClearImage}
                    disabled={isLoading}
                    externalPreview={preview}
                 />

                 <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                    <div className="flex items-center justify-between text-xs font-bold uppercase tracking-wider">
                      <span className="text-slate-500 flex items-center gap-2">
                        <Database className="w-3 h-3" />
                        Target Dataset
                      </span>
                      <span className="text-indigo-400 font-mono">MLRSNet</span>
                    </div>
                 </div>

                 {preview && !result && (
                  <button
                    onClick={handlePredict}
                    disabled={isLoading}
                    className="w-full py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 text-white font-bold transition-all shadow-xl shadow-indigo-600/20 flex items-center justify-center gap-3 active:scale-[0.95]"
                  >
                    {isLoading ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                        Running Engine...
                      </>
                    ) : (
                      <>
                        Run Recognition Engine
                        <ChevronRight className="w-4 h-4" />
                      </>
                    )}
                  </button>
                 )}

                 {result && (
                    <button
                      onClick={handleClearImage}
                      className="w-full py-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-bold transition-all border border-slate-700"
                    >
                      Process New Image
                    </button>
                 )}

                 {error && (
                    <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs text-center">
                      {error}
                    </div>
                 )}
              </CardContent>
            </Card>
          </div>

          {/* Column 2: Results & Loader */}
          <div className="lg:col-span-7">
            {isLoading ? (
              <Card className="h-full flex items-center justify-center min-h-[500px]">
                <NeuralNetworkLoader />
              </Card>
            ) : result ? (
              <PredictionResults result={result} />
            ) : (
               <Card className="h-full border-dashed border-slate-800 bg-slate-950/20 min-h-[500px] flex flex-col items-center justify-center p-12 text-center">
                 <div className="w-20 h-20 rounded-3xl bg-slate-900 border border-slate-800 flex items-center justify-center mb-6">
                    <Map className="w-10 h-10 text-slate-700" />
                 </div>
                 <h3 className="text-xl font-bold text-slate-300 mb-2">Awaiting Image</h3>
                 <p className="text-slate-500 text-sm max-w-xs">
                   Upload a satellite image to begin the AMSI-Net feature extraction process.
                 </p>
               </Card>
            )}
          </div>
        </div>

        {/* Physical Bottom: Technical Specs (Now shows at last for both mobile & desktop) */}
        <div className="pt-8">
           <div className="flex items-center gap-2 mb-4 px-4">
              <div className="h-px flex-1 bg-slate-800" />
              <span className="text-[10px] font-bold text-slate-600 uppercase tracking-[0.2em]">Architecture Specifications</span>
              <div className="h-px flex-1 bg-slate-800" />
           </div>
           <TechnicalSpecs />
        </div>

      </main>

      <footer className="mt-20 border-t border-slate-900 bg-slate-950/50 backdrop-blur-sm py-12">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-xs text-slate-500">
            AMSI-Net Land Use & Land Cover Recognition System
          </p>
        </div>
      </footer>
    </div>
  );
}
