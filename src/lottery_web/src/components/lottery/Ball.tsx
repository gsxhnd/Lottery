export function Ball(props: { num?: number; type: "red" | "blue"; size?: "sm" | "md" | "lg"; delay?: number }) {
  const { num, type, size = "md", delay = 0 } = props
  return (
    <span className={`ball ball--${type} ball--${size}`} style={{ animationDelay: `${delay}ms` }}>
      {num ? String(num).padStart(2, "0") : ""}
    </span>
  )
}
