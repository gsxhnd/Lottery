import { useEffect } from "react"
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router"

import { AppShell } from "@/components/layout/AppShell"
import { useLotteryDashboard } from "@/hooks/useLotteryDashboard"
import { AnalyticsPage } from "@/pages/AnalyticsPage"
import { DataRecordsPage } from "@/pages/DataRecordsPage"
import { PredictPage } from "@/pages/PredictPage"

function RoutedApp() {
  const state = useLotteryDashboard()
  const location = useLocation()

  useEffect(() => {
    if ((location.pathname === "/analytics" || location.pathname === "/data") && !state.vizStats && !state.vizLoading && !state.vizError) {
      void state.loadVizData(state.recentLimit)
    }
  }, [location.pathname, state])

  return (
    <>
      <Routes>
        <Route path="/" element={<AppShell healthStatus={state.healthStatus} healthText={state.healthText} />}>
          <Route index element={<Navigate to="/predict" replace />} />
          <Route
            path="predict"
            element={
              <PredictPage
                modelItems={state.modelItems}
                selectedModelPath={state.selectedModelPath}
                loadedModelPath={state.loadedModelPath}
                saveSummary={state.saveSummary}
                isLoadingModel={state.isLoadingModel}
                isPredicting={state.isPredicting}
                prediction={state.prediction}
                canPredict={state.canPredict}
                setSelectedModelPath={state.setSelectedModelPath}
                setSaveSummary={state.setSaveSummary}
                loadModels={state.loadModels}
                loadSelectedModel={state.loadSelectedModel}
                runPredict={state.runPredict}
                reloadData={state.reloadData}
              />
            }
          />
          <Route
            path="analytics"
            element={
              <AnalyticsPage
                recentLimit={state.recentLimit}
                vizLoading={state.vizLoading}
                vizStats={state.vizStats}
                setRecentLimit={state.setRecentLimit}
                loadVizData={state.loadVizData}
              />
            }
          />
          <Route
            path="data"
            element={
              <DataRecordsPage
                recentLimit={state.recentLimit}
                vizLoading={state.vizLoading}
                vizStats={state.vizStats}
                setRecentLimit={state.setRecentLimit}
                loadVizData={state.loadVizData}
              />
            }
          />
        </Route>
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <RoutedApp />
    </BrowserRouter>
  )
}
