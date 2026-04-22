'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ImageUploadZoneProps {
  onImageSelected: (file: File, previewUrl: string) => void;
  onClear: () => void;
  disabled?: boolean;
  externalPreview?: string | null;
}

export function ImageUploadZone({ onImageSelected, onClear, disabled, externalPreview }: ImageUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  // Sync with external preview (from parent)
  useEffect(() => {
    if (externalPreview !== undefined) {
      setPreview(externalPreview);
    }
  }, [externalPreview]);

  const handleFile = useCallback((file: File) => {
    if (file && file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      setPreview(url);
      onImageSelected(file, url);
    }
  }, [onImageSelected]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;

    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }, [disabled, handleFile]);

  const onChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [disabled, handleFile]);

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setPreview(null);
    onClear();
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
      className={cn(
        "relative rounded-xl border-2 border-dashed transition-all duration-200 group cursor-pointer",
        isDragging ? "border-indigo-500 bg-indigo-500/10" : "border-slate-800 hover:border-slate-700 hover:bg-slate-800/50",
        disabled && "opacity-50 cursor-not-allowed",
        preview ? "p-0 overflow-hidden aspect-video bg-black" : "p-12"
      )}
    >
      {!preview && (
        <input
          type="file"
          className="absolute inset-0 opacity-0 cursor-pointer disabled:cursor-not-allowed z-10"
          onChange={onChange}
          accept="image/*"
          disabled={disabled}
        />
      )}

      {preview ? (
        <div className="relative w-full h-full flex items-center justify-center">
          <img src={preview} alt="Upload preview" className="max-w-full max-h-full object-contain" />
          
          {/* Close Button */}
          <button
            onClick={handleClear}
            disabled={disabled}
            className="absolute top-2 right-2 p-1.5 rounded-full bg-red-500/80 hover:bg-red-500 text-white z-20 transition-all opacity-0 group-hover:opacity-100"
            title="Clear image"
          >
            <X className="w-4 h-4" />
          </button>

          <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center pointer-events-none">
            <p className="text-white text-xs font-medium">Click top-right to clear or drop new image</p>
          </div>
          
          {/* Hidden input for re-uploading via clicking on the preview (non-close button area) */}
          <input
            type="file"
            className="absolute inset-0 opacity-0 cursor-pointer z-10"
            onChange={onChange}
            accept="image/*"
            disabled={disabled}
          />
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center space-y-4 text-center">
          <div className="p-3 rounded-full bg-slate-800 text-slate-400 group-hover:text-indigo-400 transition-colors">
            <Upload className="w-8 h-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-200">
              Drop your satellite image here
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Supports JPEG, PNG up to 10MB
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
