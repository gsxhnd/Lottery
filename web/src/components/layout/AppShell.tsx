import { NavLink, Outlet } from "react-router"

import { Ball } from "@/components/lottery/Ball"

export function AppShell(props: { healthStatus: "loading" | "ok" | "error"; healthText: string }) {
  const { healthStatus, healthText } = props
  return (
    <>
      <div className="bg-grid" aria-hidden="true" />
      <div className="bg-glow bg-glow--red" aria-hidden="true" />
      <div className="bg-glow bg-glow--blue" aria-hidden="true" />
      <div className="app">
        <header className="header">
          <div className="brand">
            <div className="brand__icon" aria-hidden="true">
              <Ball type="red" size="sm" />
              <Ball type="blue" size="sm" />
            </div>
            <div>
              <h1 className="brand__title">Lottery</h1>
              <p className="brand__sub">LSTM 预测控制台</p>
            </div>
          </div>
          <div className="header__meta">
            <div className={`pill pill--${healthStatus}`}>
              <span className="pill__dot" />
              <span className="pill__text">{healthText}</span>
            </div>
            <nav className="header-nav" aria-label="main nav">
              <NavLink to="/predict" className={({ isActive }) => `btn btn--ghost ${isActive ? "btn--active" : ""}`}>
                预测
              </NavLink>
              <NavLink to="/analytics" className={({ isActive }) => `btn btn--ghost ${isActive ? "btn--active" : ""}`}>
                可视化
              </NavLink>
              <NavLink to="/data" className={({ isActive }) => `btn btn--ghost ${isActive ? "btn--active" : ""}`}>
                数据
              </NavLink>
            </nav>
            <a className="link-api" href="/docs" target="_blank" rel="noreferrer">
              API 文档
            </a>
          </div>
        </header>
        <Outlet />
      </div>
    </>
  )
}
