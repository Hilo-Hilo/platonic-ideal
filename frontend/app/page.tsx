"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Sparkles, Plus, X, ChevronDown, ChevronUp, GripVertical, Settings } from "lucide-react";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

interface Group {
  id: string;
  name: string;
  weight: number;
  entries: string[];
}

interface TopWord {
  word: string;
  cosine_similarity: number;
  token_ids: number[];
  token_pieces?: string[];
}

interface ComputeResult {
  _output: {
    model: {
      model_id: string;
      repo_id: string;
      embedding_dim: number;
      vocab_size: number;
    };
    settings: Record<string, unknown>;
    groups: Array<Record<string, unknown>>;
    top_words: TopWord[];
    overall: {
      norm: number;
    };
    wordnet_stats?: Record<string, unknown>;
  };
}

interface ModelInfo {
  id: string;
  name: string;
  repo_id: string;
  speed: string;
  description: string;
}

const DEFAULT_GROUP: Group = {
  id: "default-1",
  name: "space",
  weight: 1.0,
  entries: ["planet", "star", "sun", "earth", "galaxy", "solar system", "orbit", "astronomy"]
};

// Sortable Group Card Component
function SortableGroupCard({
  group,
  groups,
  updateGroupName,
  updateGroupWeight,
  removeGroup,
  removeEntry,
  newEntry,
  setNewEntry,
  addEntry,
}: {
  group: Group;
  groups: Group[];
  updateGroupName: (id: string, name: string) => void;
  updateGroupWeight: (id: string, weight: number) => void;
  removeGroup: (id: string) => void;
  removeEntry: (id: string, idx: number) => void;
  newEntry: {[key: string]: string};
  setNewEntry: (e: {[key: string]: string}) => void;
  addEntry: (id: string) => void;
}) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: group.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div ref={setNodeRef} style={style}>
      <Card className="border-2">
        <CardContent className="pt-4 space-y-3">
          {/* Group Header with Drag Handle */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="cursor-grab active:cursor-grabbing touch-none"
              {...attributes}
              {...listeners}
            >
              <GripVertical className="h-5 w-5 text-slate-400" />
            </button>
            <Input
              placeholder="Group name (optional)"
              value={group.name}
              onChange={(e) => updateGroupName(group.id, e.target.value)}
              className="flex-1"
            />
            {groups.length > 1 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => removeGroup(group.id)}
                type="button"
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>

          {/* Weight Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-600 dark:text-slate-400">Weight:</span>
              <span className="font-mono">{group.weight.toFixed(2)}</span>
            </div>
            <Slider
              value={[group.weight]}
              onValueChange={(v) => updateGroupWeight(group.id, v[0])}
              min={-2}
              max={2}
              step={0.1}
              className="cursor-pointer"
            />
          </div>

          {/* Entries */}
          <div className="space-y-2">
            <span className="text-sm text-slate-600 dark:text-slate-400">Entries:</span>
            <div className="flex flex-wrap gap-2">
              {group.entries.map((entry, idx) => (
                <Badge 
                  key={idx} 
                  variant="secondary" 
                  className="group flex items-center gap-1"
                >
                  <span>{entry}</span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeEntry(group.id, idx);
                    }}
                    className="opacity-60 hover:opacity-100 transition-opacity"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </Badge>
              ))}
            </div>
            
            {/* Add Entry */}
            <div className="flex gap-2">
              <Input
                placeholder="Add word or phrase..."
                value={newEntry[group.id] || ""}
                onChange={(e) => setNewEntry({ ...newEntry, [group.id]: e.target.value })}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addEntry(group.id);
                  }
                }}
                className="flex-1"
              />
              <Button
                size="icon"
                variant="outline"
                onClick={() => addEntry(group.id)}
                type="button"
              >
                <Plus className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default function Home() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [groups, setGroups] = useState<Group[]>([DEFAULT_GROUP]);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [topK, setTopK] = useState<number>(20);
  const [results, setResults] = useState<Map<string, ComputeResult>>(new Map());
  const [activeResultTab, setActiveResultTab] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [showExplanation, setShowExplanation] = useState(false);
  const [showOutputDetails, setShowOutputDetails] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [newEntry, setNewEntry] = useState<{[groupId: string]: string}>({});
  const [mounted, setMounted] = useState(false);

  // Drag and drop sensors
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      const oldIndex = groups.findIndex((g) => g.id === active.id);
      const newIndex = groups.findIndex((g) => g.id === over.id);

      setGroups(arrayMove(groups, oldIndex, newIndex));
    }
  };

  // Show onboarding on first visit
  useEffect(() => {
    setMounted(true);
    // Session ID shared across tabs via localStorage
    let sid = localStorage.getItem("platonic_session_id");
    if (!sid) {
      sid = (globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`).toString();
      localStorage.setItem("platonic_session_id", sid);
    }
    setSessionId(sid);
    const hasSeenOnboarding = localStorage.getItem("hasSeenOnboarding");
    if (!hasSeenOnboarding) {
      setShowOnboarding(true);
      localStorage.setItem("hasSeenOnboarding", "true");
    }

    // Fetch allowed models from backend
    const apiBaseUrl =
      process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

    fetch(`${apiBaseUrl}/health`)
      .then(res => res.json())
      .then(data => {
        if (data.available_models) {
          setModels(data.available_models);
          // Auto-select default model if valid
          if (data.default_model_id && data.available_models.some((m: ModelInfo) => m.id === data.default_model_id)) {
            setSelectedModels(new Set([data.default_model_id]));
            setActiveResultTab(data.default_model_id);
          } else if (data.available_models.length > 0) {
            setSelectedModels(new Set([data.available_models[0].id]));
            setActiveResultTab(data.available_models[0].id);
          }
        }
      })
      .catch(err => {
        console.error("Failed to fetch models:", err);
        setError("Could not load available models from server.");
      });

  }, []);

  const addGroup = () => {
    const newGroup: Group = {
      id: `group-${Date.now()}`,
      name: "",
      weight: 1.0,
      entries: []
    };
    setGroups([...groups, newGroup]);
  };

  const removeGroup = (groupId: string) => {
    setGroups(groups.filter(g => g.id !== groupId));
  };

  const updateGroupName = (groupId: string, name: string) => {
    setGroups(groups.map(g => g.id === groupId ? { ...g, name } : g));
  };

  const updateGroupWeight = (groupId: string, weight: number) => {
    setGroups(groups.map(g => g.id === groupId ? { ...g, weight } : g));
  };

  const addEntry = (groupId: string) => {
    const entry = newEntry[groupId]?.trim();
    if (!entry) return;
    
    setGroups(groups.map(g => 
      g.id === groupId ? { ...g, entries: [...g.entries, entry] } : g
    ));
    setNewEntry({ ...newEntry, [groupId]: "" });
  };

  const removeEntry = (groupId: string, entryIndex: number) => {
    setGroups(groups.map(g =>
      g.id === groupId 
        ? { ...g, entries: g.entries.filter((_, i) => i !== entryIndex) }
        : g
    ));
  };

  const toggleModelSelection = (modelId: string) => {
    const newSet = new Set(selectedModels);
    if (newSet.has(modelId)) {
      if (newSet.size > 1) { // Keep at least one selected
        newSet.delete(modelId);
      }
    } else {
      if (newSet.size >= 3) {
        setError("You can select at most 3 models per request.");
        return;
      }
      newSet.add(modelId);
    }
    setSelectedModels(newSet);
  };

  const analyze = async () => {
    // Validate input
    const validGroups = groups.filter(g => g.entries.length > 0);
    if (validGroups.length === 0) {
      setError("Please add at least one entry to a group");
      return;
    }

    if (selectedModels.size === 0) {
      setError("Please select at least one model");
      return;
    }
    if (selectedModels.size > 3) {
      setError("You can select at most 3 models per request.");
      return;
    }
    if (!sessionId) {
      setError("Session not ready yet. Please refresh the page.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(new Map());

    const apiBaseUrl =
      process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || "http://127.0.0.1:8000";

    const payload = {
      model_ids: Array.from(selectedModels),
      groups: validGroups.map(g => ({
        name: g.name || undefined,
        weight: g.weight,
        entries: g.entries
      })),
      options: {
        top_k: topK,
        wordnet_pos: "n,v",
        exclude_input: true,
        exclude_substrings: true,
        min_word_chars: 3,
        max_token_len: 6
      }
    };

    try {
      const response = await fetch(`${apiBaseUrl}/compute-essence`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-Session-ID": sessionId,
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(120000),
      });

      if (response.status === 429) {
        const errData = await response.json().catch(() => ({ detail: "Rate limited" }));
        throw new Error(errData.detail || "Another request is already running in a different tab.");
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }

      const data = await response.json();
      const newResults = new Map<string, ComputeResult>();
      const errors: string[] = [];

      const resultObj = data?.results ?? {};
      for (const [modelId, modelResult] of Object.entries(resultObj)) {
        newResults.set(modelId, modelResult as ComputeResult);
      }
      const errorObj = data?.errors ?? {};
      for (const [modelId, msg] of Object.entries(errorObj)) {
        errors.push(`${modelId}: ${msg}`);
      }

      setResults(newResults);
      if (newResults.size > 0) {
        setActiveResultTab(Array.from(newResults.keys())[0]);
      }
      if (errors.length > 0) {
        setError(errors.join("\n"));
      }
    } catch (err) {
      if (err instanceof Error) {
        if (err.name === "AbortError") {
          setError("Request timed out. Try fewer models or smaller models.");
        } else if (err.message.includes("Failed to fetch") || err.message.includes("ERR_CONNECTION_REFUSED")) {
          setError(
            `Cannot connect to backend. Make sure the API server is running on ${apiBaseUrl} (if you used localhost, try 127.0.0.1 to avoid IPv6 localhost issues).`
          );
        } else {
          setError(err.message);
        }
      } else {
        setError("Unknown error");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      {/* Onboarding Modal */}
      <Dialog open={showOnboarding} onOpenChange={setShowOnboarding}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <Sparkles className="h-6 w-6 text-yellow-500" />
              Welcome to Platonic Ideal
            </DialogTitle>
            <DialogDescription className="space-y-4 text-base pt-4">
              <p>
                <strong>Platonic Ideal</strong> lets you explore the semantic essence of word groups using language model embeddings.
              </p>
              
              <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-lg space-y-2">
                <p className="font-semibold">How it works:</p>
                <ol className="list-decimal list-inside space-y-1 text-sm">
                  <li>Create <strong>word groups</strong> with related concepts (e.g., &ldquo;space&rdquo;, &ldquo;ocean&rdquo;)</li>
                  <li>Assign <strong>weights</strong> to emphasize or de-emphasize each group</li>
                  <li>Click <strong>Analyze</strong> to compute the mathematical &ldquo;essence&rdquo; vector</li>
                  <li>See which <strong>dictionary words</strong> are most aligned with your essence</li>
                </ol>
              </div>

              <p className="text-sm">
                The algorithm uses non-contextual token embeddings from language models and ranks 
                ~60,000 WordNet dictionary words by cosine similarity.
              </p>
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end">
            <Button onClick={() => setShowOnboarding(false)}>
              Get Started
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Main Page */}
      <div className="container mx-auto p-6">
        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <Sparkles className="h-8 w-8 text-yellow-500" />
            Platonic Ideal
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Compute semantic essence vectors from word groups
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left: Input Section */}
          <div className="lg:col-span-1">
            <Card className="h-[calc(100vh-20rem)]">
              <CardHeader>
                <CardTitle>Input Groups</CardTitle>
                <CardDescription>
                  Define word groups with configurable weights
                </CardDescription>
              </CardHeader>
              <ScrollArea className="h-[calc(100%-5rem)]">
                <CardContent className="space-y-4 pr-4">
                  {mounted ? (
                    <DndContext
                      sensors={sensors}
                      collisionDetection={closestCenter}
                      onDragEnd={handleDragEnd}
                    >
                      <SortableContext
                        items={groups.map(g => g.id)}
                        strategy={verticalListSortingStrategy}
                      >
                        {groups.map((group) => (
                          <SortableGroupCard
                            key={group.id}
                            group={group}
                            groups={groups}
                            updateGroupName={updateGroupName}
                            updateGroupWeight={updateGroupWeight}
                            removeGroup={removeGroup}
                            removeEntry={removeEntry}
                            newEntry={newEntry}
                            setNewEntry={setNewEntry}
                            addEntry={addEntry}
                          />
                        ))}
                      </SortableContext>
                    </DndContext>
                  ) : (
                    <div className="space-y-4">
                      {groups.map((group) => (
                        <Card key={group.id} className="border-2">
                          <CardContent className="pt-4">
                            <div className="animate-pulse space-y-3">
                              <div className="h-10 bg-slate-200 dark:bg-slate-700 rounded"></div>
                              <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded"></div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}

                  <Button onClick={addGroup} variant="outline" className="w-full">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Group
                  </Button>
                </CardContent>
              </ScrollArea>
            </Card>
          </div>

          {/* Center: Analyze Section */}
          <div className="lg:col-span-1 flex flex-col items-center justify-start pt-8 space-y-4">
            <Button
              size="lg"
              onClick={analyze}
              disabled={loading}
              className="w-full max-w-xs h-16 text-lg"
            >
              {loading ? "Computing..." : "Analyze"}
            </Button>

            {/* Config Section */}
            <Card className="w-full max-w-xs">
              <CardHeader 
                className="cursor-pointer py-3"
                onClick={() => setShowConfig(!showConfig)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Settings className="h-4 w-4" />
                    <span className="text-sm font-medium">Configuration</span>
                  </div>
                  {showConfig ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </div>
              </CardHeader>
              {showConfig && (
                <CardContent className="pt-0 space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm text-slate-600 dark:text-slate-400">
                      Top K Words: <span className="font-mono">{topK}</span>
                    </label>
                    <Slider
                      value={[topK]}
                      onValueChange={(v) => setTopK(v[0])}
                      min={5}
                      max={50}
                      step={5}
                      className="cursor-pointer"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <label className="text-sm text-slate-600 dark:text-slate-400">
                      Models to Analyze
                    </label>
                    <div className="space-y-3">
                      {models.length === 0 ? (
                         <div className="text-sm text-slate-500 italic">Loading models...</div>
                      ) : (
                        models.map((model) => (
                        <div key={model.id} className="flex items-start space-x-3">
                          <Checkbox
                            id={`model-${model.id}`}
                            checked={selectedModels.has(model.id)}
                            onCheckedChange={() => toggleModelSelection(model.id)}
                            className="mt-1"
                          />
                          <label
                            htmlFor={`model-${model.id}`}
                            className="text-sm cursor-pointer flex-1 leading-tight peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                          >
                            <div>{model.name}</div>
                            <div className="text-xs text-slate-500 mt-0.5">
                              Speed: {model.speed}
                            </div>
                          </label>
                        </div>
                      )))}
                    </div>
                    <p className="text-xs text-slate-500">
                      {selectedModels.size} model{selectedModels.size !== 1 ? "s" : ""} selected
                    </p>
                  </div>
                </CardContent>
              )}
            </Card>

            {error && (
              <div className="text-red-500 text-sm p-4 bg-red-50 dark:bg-red-950 rounded w-full max-w-xs whitespace-pre-line">
                {error}
              </div>
            )}
          </div>

          {/* Right: Output Section */}
          <div className="lg:col-span-1">
            <Card className="h-[calc(100vh-20rem)]">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Results</CardTitle>
                    <CardDescription>
                      Top dictionary words aligned with essence
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              {results.size > 0 && (
                <div className="px-6 pb-3">
                  <Tabs value={activeResultTab} onValueChange={setActiveResultTab}>
                    <TabsList className="w-full flex">
                      {Array.from(results.keys()).map((modelId) => {
                        const model = models.find(m => m.id === modelId);
                        const modelName = model?.name || modelId;
                        // Extract first part for tab label
                        const shortName = modelName.split(" ")[0];
                        return (
                          <TabsTrigger key={modelId} value={modelId} className="text-xs px-3 flex-1">
                            {shortName}
                          </TabsTrigger>
                        );
                      })}
                    </TabsList>
                  </Tabs>
                </div>
              )}

              <ScrollArea className="h-[calc(100%-12rem)]">
                <div className="px-6 py-4">
                  {results.size > 0 ? (
                    Array.from(results.entries()).map(([modelId, result]) => (
                      activeResultTab === modelId && (
                        <div key={modelId} className="space-y-3">
                          {/* Model Info Dropdown */}
                          <div className="mb-3">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setShowOutputDetails(!showOutputDetails)}
                            className="w-full justify-between"
                          >
                            <span className="text-xs">Model Information</span>
                            {showOutputDetails ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </Button>
                          
                          {showOutputDetails && (
                            <div className="mt-2 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg text-xs space-y-1">
                              <div>Model: <span className="font-mono">{result._output.model.model_id}</span></div>
                              <div>Repo: <span className="font-mono text-[10px]">{result._output.model.repo_id}</span></div>
                              <div>Vocab: <span className="font-mono">{result._output.model.vocab_size.toLocaleString()}</span></div>
                              <div>Dim: <span className="font-mono">{result._output.model.embedding_dim}</span></div>
                              <div>Norm: <span className="font-mono">{result._output.overall.norm.toFixed(4)}</span></div>
                            </div>
                          )}
                        </div>

                        {/* Top Words List */}
                        <div className="space-y-2">
                          {result._output.top_words.map((w, idx) => (
                            <div
                              key={idx}
                              className="flex items-center justify-between p-3 rounded-lg bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors"
                            >
                              <div className="flex items-center gap-3 min-w-0">
                                <span className="text-sm font-mono text-slate-500 dark:text-slate-400 w-8 flex-shrink-0">
                                  {idx + 1}.
                                </span>
                                <span className="font-medium truncate">{w.word}</span>
                              </div>
                              <span className="font-mono text-sm text-slate-600 dark:text-slate-400 flex-shrink-0">
                                {w.cosine_similarity.toFixed(4)}
                              </span>
                            </div>
                          ))}
                        </div>
                        </div>
                      )
                    ))
                  ) : !loading && !error ? (
                    <div className="text-center py-12 text-slate-400">
                      <Sparkles className="h-12 w-12 mx-auto mb-3 opacity-50" />
                      <p>Results will appear here</p>
                      <p className="text-xs mt-2">Select models and click Analyze</p>
                    </div>
                  ) : loading ? (
                    <div className="text-center py-12">
                      <div className="animate-spin h-8 w-8 border-4 border-slate-300 border-t-slate-600 rounded-full mx-auto mb-3"></div>
                      <p className="text-slate-600 dark:text-slate-400">Computing {selectedModels.size} model{selectedModels.size > 1 ? "s" : ""}...</p>
                      <p className="text-xs text-slate-500 mt-2">This may take 1-2 minutes per model on first run</p>
                    </div>
                  ) : null}
                </div>
              </ScrollArea>
            </Card>
          </div>
        </div>

        {/* Bottom: Explanation Section */}
        <Card className="mt-6">
          <CardHeader className="cursor-pointer" onClick={() => setShowExplanation(!showExplanation)}>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">How Does This Work?</CardTitle>
              {showExplanation ? <ChevronUp className="h-5 w-5" /> : <ChevronDown className="h-5 w-5" />}
            </div>
          </CardHeader>
          {showExplanation && (
            <CardContent className="space-y-4 text-sm text-slate-700 dark:text-slate-300">
              <div className="space-y-3">
                <h3 className="font-semibold text-base">The Simple Explanation</h3>
                <p>
                  Every word in a language model has a <strong>vector</strong> (a list of numbers) that represents its meaning. 
                  Words with similar meanings have vectors that point in similar directions in a high-dimensional space.
                </p>
                <p>
                  This tool lets you create groups of related words, combine them mathematically, and discover 
                  which dictionary words are most aligned with that combination.
                </p>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold text-base">What Are Weights?</h3>
                <p>
                  <strong>Positive weights</strong> (like 1.0) pull the result <em>toward</em> that group&apos;s meaning.
                </p>
                <p>
                  <strong>Negative weights</strong> (like -0.5) push the result <em>away</em> from that group.
                </p>
                <p>
                  <strong>Example</strong>: If you have &ldquo;nature&rdquo; (weight: 1.0) and &ldquo;urban&rdquo; (weight: -0.5), 
                  the result will favor wilderness-related words and avoid city-related words.
                </p>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold text-base">The Math (For the Curious)</h3>
                <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded font-mono text-xs space-y-2">
                  <p>1. Each entry → mean(token_vectors) = entry_vector</p>
                  <p>2. Each group → mean(entry_vectors) = group_mean</p>
                  <p>3. Apply weights → weight × group_mean = weighted_group</p>
                  <p>4. Overall essence → mean(all weighted_groups)</p>
                  <p>5. Rank ~60k WordNet words by cosine_similarity(word_vec, essence_vec)</p>
                </div>
                <p className="text-xs text-slate-500">
                  This uses non-contextual token embeddings. 
                  See the full mathematical details in the 
                  <a 
                    href="https://github.com/Hilo-Hilo/platonic-ideal" 
                    target="_blank"
                    className="underline hover:text-blue-600"
                  >
                    README
                  </a>.
                </p>
              </div>

              <div className="mt-4 pt-4 border-t">
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setShowOnboarding(true)}
                >
                  Show Tutorial Again
                </Button>
              </div>
            </CardContent>
          )}
        </Card>
      </div>
    </div>
  );
}
