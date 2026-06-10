import { useState, useEffect } from 'react'
import { BarChart3, AlertTriangle, TrendingUp, Lightbulb } from 'lucide-react'
import api from '../api'

export default function Analytics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchAnalytics()
  }, [])

  const fetchAnalytics = async () => {
    try {
      const { data: resp } = await api.get('/analytics/hallucinations')
      if (resp.success) {
        setData(resp.data)
      } else {
        setError(resp.error || 'Failed to load analytics')
      }
    } catch (err) {
      if (err.response?.status === 403) {
        setError('Admin access required for analytics')
      } else {
        setError(err.message || 'Connection failed')
      }
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <span className="text-gray-500 animate-pulse">Loading analytics...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold text-gray-900 mb-6">Hallucination Analytics</h1>
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
          {error}
        </div>
      </div>
    )
  }

  const summary = data?.summary || {}
  const byIntent = data?.by_intent || []
  const insights = data?.insights || []
  const recent = data?.recent_hallucinations || []

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Hallucination Analytics</h1>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard
          label="Total Queries"
          value={summary.total_queries || 0}
          icon={BarChart3}
        />
        <MetricCard
          label="Hallucinations"
          value={summary.total_hallucinations || 0}
          icon={AlertTriangle}
          color="red"
        />
        <MetricCard
          label="Hallucination Rate"
          value={`${((summary.hallucination_rate || 0) * 100).toFixed(1)}%`}
          icon={TrendingUp}
          color={summary.hallucination_rate > 0.1 ? 'red' : 'green'}
        />
        <MetricCard
          label="Avg Latency"
          value={`${Math.round(summary.avg_latency_ms || 0)}ms`}
          icon={BarChart3}
        />
      </div>

      {/* Insights */}
      {insights.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 mb-8">
          <div className="flex items-center gap-2 mb-3">
            <Lightbulb size={18} className="text-amber-600" />
            <h3 className="font-medium text-amber-900 text-sm">Insights</h3>
          </div>
          <ul className="space-y-2">
            {insights.map((insight, i) => (
              <li key={i} className="text-sm text-amber-800">{insight}</li>
            ))}
          </ul>
        </div>
      )}

      {/* By Intent */}
      {byIntent.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm mb-8">
          <h3 className="font-medium text-gray-900 mb-4">Hallucination Rate by Intent</h3>
          <div className="space-y-3">
            {byIntent.map((item) => (
              <div key={item.intent} className="flex items-center gap-4">
                <span className="text-sm font-medium text-gray-700 w-28">{item.intent}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      item.hallucination_rate > 0.2 ? 'bg-red-500' :
                      item.hallucination_rate > 0.1 ? 'bg-amber-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${Math.max(item.hallucination_rate * 100, 2)}%` }}
                  />
                </div>
                <span className="text-xs text-gray-500 w-20 text-right">
                  {(item.hallucination_rate * 100).toFixed(1)}% ({item.hallucinations}/{item.total_queries})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Hallucinations */}
      {recent.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
          <h3 className="font-medium text-gray-900 mb-4">Recent Hallucinations</h3>
          <div className="space-y-3">
            {recent.map((item, i) => (
              <div key={i} className="p-3 bg-red-50 border border-red-100 rounded-lg">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-medium text-red-800">{item.intent}</span>
                  <span className="text-xs text-gray-400">{item.latency_ms}ms</span>
                  <span className="text-xs text-gray-400 ml-auto">{item.timestamp?.slice(0, 10)}</span>
                </div>
                <p className="text-sm text-gray-800 font-medium">{item.query}</p>
                {item.answer_preview && (
                  <p className="text-xs text-gray-600 mt-1 line-clamp-2">{item.answer_preview}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {summary.total_queries === 0 && (
        <div className="text-center py-16 text-gray-500">
          <BarChart3 size={48} className="mx-auto mb-4 opacity-30" />
          <p className="text-lg">No queries yet</p>
          <p className="text-sm mt-1">Run some queries to see hallucination analytics here</p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, icon: Icon, color = 'indigo' }) {
  const colors = {
    indigo: 'text-indigo-600 bg-indigo-50 border-indigo-200',
    red: 'text-red-600 bg-red-50 border-red-200',
    green: 'text-green-600 bg-green-50 border-green-200',
  }

  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <Icon size={18} className="mb-2 opacity-70" />
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs opacity-70">{label}</div>
    </div>
  )
}
