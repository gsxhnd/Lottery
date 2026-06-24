import { useEffect, useMemo, useState } from "react"
import { toast } from "sonner"

import { api } from "@/lib/api"
import type { HealthResponse, ModelInfo, PredictionResponse, WinningStatsResponse } from "@/types/lottery"

export function useLotteryDashboard() {
  const [healthStatus, setHealthStatus] = useState<"loading" | "ok" | "error">("loading")
  const [healthText, setHealthText] = useState("连接中…")
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModelPath, setSelectedModelPath] = useState<string | null>(null)
  const [loadedModelPath, setLoadedModelPath] = useState<string | null>(null)
  const [saveSummary, setSaveSummary] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [vizStats, setVizStats] = useState<WinningStatsResponse | null>(null)
  const [vizLoading, setVizLoading] = useState(false)
  const [vizError, setVizError] = useState<string | null>(null)
  const [recentLimit, setRecentLimit] = useState(120)

  const modelItems = Array.isArray(models) ? models : []
  const canPredict = useMemo(() => Boolean(loadedModelPath || selectedModelPath) && !isPredicting, [isPredicting, loadedModelPath, selectedModelPath])

  const showToast = (message: string, isError = false) => {
    if (isError) {
      toast.error(message)
      return
    }
    toast.success(message)
  }

  const refreshHealth = async () => {
    try {
      const h = await api<HealthResponse>("/health")
      const modelNote = h.model_loaded ? " · 模型已加载" : ""
      setHealthStatus("ok")
      setHealthText(`${h.records.toLocaleString()} 期数据${modelNote}`)
    } catch {
      setHealthStatus("error")
      setHealthText("API 不可用")
    }
  }

  const fetchCurrentModel = async () => {
    try {
      const current = await api<ModelInfo>("/models/current")
      setLoadedModelPath(current.path)
      setSelectedModelPath((v) => v ?? current.path)
    } catch {
      setLoadedModelPath(null)
    }
  }

  const loadModels = async () => {
    try {
      const raw = await api<unknown>("/models")
      const list = Array.isArray(raw) ? (raw as ModelInfo[]) : []
      setModels(list)
      setSelectedModelPath((current) => current ?? list[0]?.path ?? null)
    } catch (error) {
      setModels([])
      showToast(error instanceof Error ? error.message : "加载模型列表失败", true)
    }
  }

  const loadSelectedModel = async () => {
    if (!selectedModelPath) return
    setIsLoadingModel(true)
    try {
      const info = await api<ModelInfo>("/models/load", { method: "POST", body: JSON.stringify({ model: selectedModelPath }) })
      setLoadedModelPath(info.path)
      showToast("模型已加载")
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "加载模型失败", true)
    } finally {
      setIsLoadingModel(false)
    }
  }

  const runPredict = async () => {
    setIsPredicting(true)
    try {
      const body: { save_summary: boolean; model?: string } = { save_summary: saveSummary }
      if (!loadedModelPath && selectedModelPath) body.model = selectedModelPath
      const data = await api<PredictionResponse>("/predict", { method: "POST", body: JSON.stringify(body) })
      setPrediction(data)
      if (data.model_dir) setLoadedModelPath(data.model_dir)
      showToast("预测完成")
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "预测失败", true)
    } finally {
      setIsPredicting(false)
    }
  }

  const reloadData = async () => {
    try {
      const data = await api<{ records: number }>("/data/reload", { method: "POST" })
      showToast(`已刷新 ${data.records.toLocaleString()} 期数据`)
      await refreshHealth()
    } catch (error) {
      showToast(error instanceof Error ? error.message : "刷新失败", true)
    }
  }

  const loadVizData = async (limit: number) => {
    setVizLoading(true)
    setVizError(null)
    try {
      const data = await api<WinningStatsResponse>(`/data/winning-stats?recent_limit=${limit}`)
      setVizStats(data)
    } catch (error) {
      const message = error instanceof Error ? error.message : "加载可视化数据失败"
      setVizError(message)
      showToast(message, true)
    } finally {
      setVizLoading(false)
    }
  }

  useEffect(() => {
    void (async () => {
      await refreshHealth()
      await Promise.all([loadModels(), fetchCurrentModel()])
    })()
  }, [])

  return {
    healthStatus,
    healthText,
    modelItems,
    selectedModelPath,
    loadedModelPath,
    saveSummary,
    isLoadingModel,
    isPredicting,
    prediction,
    vizStats,
    vizLoading,
    vizError,
    recentLimit,
    canPredict,
    setSelectedModelPath,
    setSaveSummary,
    setRecentLimit,
    loadModels,
    loadSelectedModel,
    runPredict,
    reloadData,
    loadVizData,
  }
}
