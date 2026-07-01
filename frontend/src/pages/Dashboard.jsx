import { useState, useEffect } from 'react'
import { Activity, Shield, Brain, Database, Search, Upload, Zap, TrendingUp, Clock, CheckCircle } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import api, { isDemoMode } from '../api'
import { MOCK_HEALTH, MOCK_ANALYTICS } from '../demo'

export default function Dashboard() {
  const navigate = useNavigate()
  const [health, setHealth] = useState(null)
  const [stats, setStats] = useState(null)
  const user = JSON.parse(localStorage.getItem('user') || '{}')

  useEffect(() => {
    if (isDemoMode()) {
      setHealth(MOCK_HEALTH)
      setStats(MOCK_ANALYTICS.summary)
      return
    }
    api.get('/health').then(({ data }) => {
      if (data.success) setHealth(data.data)
    }).catch(() => {})

    // Fetch analytics summary if admin
    if (user.role === 'admin') {
      api.get('/analytics/hallucinations').then(({ data }) => {
        if (data.success) setStats(data.data.summary)
      }).catch(() => {})
    }
  }, [])

  const services = health?.services || {}

  return (
    <div className="max-w-5xl mx-auto">
      {/* Hero Section */}
      <div className="bg-gradient-to-br from-indigo-600 via-purple-600 to-indigo-800 rounded-2xl p-8 mb-8 text-white relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/5 rounded-full translate-y-1/2 -translate-x-1/2" />
        <div className="relative">
          <p className="text-indigo-200 text-sm font-medium mb-1">Welcome back</p>
          <h1 className="text-3xl font-bold mb-2">{user.email}</h1>
          <p className="text-indigo-200 text-sm">
            <span className="inline-flex items-center gap-1 bg-white/10 px-2 py-0.5 rounded-full text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
              {user.role}
            </span>
            <span className="mx-2">·</span>
            Tenant: {user.tenant_name}
            <span className="mx-2">·</span>
            v{health?.version || '2.0.0'}
          </p>
        </div>
      </div>

      {/* System Status Strip */}
      <div className="grid grid-cols-3 gap-3 mb-8">
        <ServicePill label="API" status={services.api} />
        <ServicePill label="PostgreSQL" status={services.postgres} />
        <ServicePill label="Qdrant" status={services.qdrant} />
      </div>

      {/* Quick Stats (if admin) */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <StatCard
            icon={Search}
            label="Total Queries"
            value={stats.total_queries}
            color="indigo"
          />
          <StatCard
            icon={CheckCircle}
            label="Clean Answers"
            value={stats.clean_answers}
            color="green"
          />
          <StatCard
            icon={Shield}
            label="Hallucinations"
            value={stats.total_hallucinations}
            color="red"
            subtitle={`${((stats.hallucination_rate || 0) * 100).toFixed(1)}% rate`}
          />
          <StatCard
            icon={Clock}
            label="Avg Latency"
            value={`${Math.round(stats.avg_latency_ms || 0)}ms`}
            color="amber"
          />
        </div>
      )}

      {/* Quick Actions */}
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <ActionCard
          icon={Search}
          title="Query Pipeline"
          description="Ask a question — intent classification, retrieval, reranking, generation, and hallucination check"
          action="Try a Query"
          onClick={() => navigate('/query')}
          color="indigo"
        />
        <ActionCard
          icon={Upload}
          title="Ingest Documents"
          description="Upload PDF, TXT, or DOCX files to expand the knowledge base"
          action="Upload Document"
          onClick={() => navigate('/ingest')}
          color="purple"
        />
        <ActionCard
          icon={TrendingUp}
          title="Analytics"
          description="View hallucination patterns, intent breakdowns, and system insights"
          action="View Analytics"
          onClick={() => navigate('/analytics')}
          color="amber"
        />
      </div>

      {/* Pipeline Architecture */}
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Pipeline Architecture</h2>
      <div className="bg-white rounded-2xl border border-gray-200 p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-center gap-2 text-xs">
          <PipelineStep icon={Search} label="Query" active />
          <Arrow />
          <PipelineStep icon={Brain} label="Intent Classifier" sublabel="RoBERTa" />
          <Arrow />
          <PipelineStep icon={Database} label="Retrieval" sublabel="Qdrant + LlamaIndex" />
          <Arrow />
          <PipelineStep icon={TrendingUp} label="Rerank" sublabel="Cohere v3.5" />
          <Arrow />
          <PipelineStep icon={Zap} label="Generate" sublabel="Groq LLM" />
          <Arrow />
          <PipelineStep icon={Shield} label="Hallucination Check" sublabel="3-method" />
          <Arrow />
          <PipelineStep icon={Activity} label="Self-Heal" sublabel="Auto-retry" />
        </div>
      </div>

      {/* Capabilities Grid */}
      <h2 className="text-lg font-semibold text-gray-900 mt-8 mb-4">Capabilities</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <CapabilityCard
          emoji="🎯"
          title="Intent-Aware Routing"
          description="6 intent categories — each routes to specialized vector indexes for targeted retrieval"
        />
        <CapabilityCard
          emoji="🔒"
          title="Multi-Tenant Isolation"
          description="Tenant data is strictly separated. One tenant never sees another's documents or queries"
        />
        <CapabilityCard
          emoji="🧠"
          title="Self-Healing"
          description="Detects hallucinations and auto-retries with corrective strategies (expand, strict, broaden)"
        />
        <CapabilityCard
          emoji="🌐"
          title="Web Fallback"
          description="When the knowledge base doesn't have an answer, automatically searches the web"
        />
        <CapabilityCard
          emoji="📊"
          title="RAGAS Evaluation"
          description="Upload test sets and get faithfulness, relevancy, precision, and recall scores"
        />
        <CapabilityCard
          emoji="🔍"
          title="Observability"
          description="Every pipeline step traced with latency, tokens, and cost — powered by LangFuse"
        />
      </div>
    </div>
  )
}

function ServicePill({ label, status }) {
  const isUp = status === 'up' || status === 'healthy'
  return (
    <div className={`flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border text-sm font-medium ${
      isUp ? 'bg-green-50 border-green-200 text-green-700' : 'bg-red-50 border-red-200 text-red-700'
    }`}>
      <span className={`w-2 h-2 rounded-full ${isUp ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
      {label}: {status || 'unknown'}
    </div>
  )
}

function StatCard({ icon: Icon, label, value, color, subtitle }) {
  const colors = {
    indigo: 'bg-indigo-50 border-indigo-200 text-indigo-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    red: 'bg-red-50 border-red-200 text-red-700',
    amber: 'bg-amber-50 border-amber-200 text-amber-700',
  }
  return (
    <div className={`rounded-xl border p-4 ${colors[color]}`}>
      <div className="flex items-center justify-between mb-2">
        <Icon size={18} className="opacity-70" />
        {subtitle && <span className="text-xs opacity-70">{subtitle}</span>}
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs opacity-70 mt-0.5">{label}</div>
    </div>
  )
}

function ActionCard({ icon: Icon, title, description, action, onClick, color }) {
  const borderColors = {
    indigo: 'hover:border-indigo-300',
    purple: 'hover:border-purple-300',
    amber: 'hover:border-amber-300',
  }
  const btnColors = {
    indigo: 'bg-indigo-600 hover:bg-indigo-700',
    purple: 'bg-purple-600 hover:bg-purple-700',
    amber: 'bg-amber-600 hover:bg-amber-700',
  }
  return (
    <div className={`bg-white rounded-xl border border-gray-200 p-5 shadow-sm transition-all ${borderColors[color]} hover:shadow-md flex flex-col`}>
      <Icon size={24} className="text-gray-600 mb-3" />
      <h3 className="font-semibold text-gray-900 text-sm mb-1">{title}</h3>
      <p className="text-xs text-gray-500 leading-relaxed flex-1 mb-4">{description}</p>
      <button
        onClick={onClick}
        className={`w-full text-white text-xs font-medium py-2 rounded-lg transition-colors ${btnColors[color]}`}
      >
        {action} →
      </button>
    </div>
  )
}

function PipelineStep({ icon: Icon, label, sublabel, active }) {
  return (
    <div className={`flex flex-col items-center gap-1 px-3 py-2 rounded-lg ${active ? 'bg-indigo-50 border border-indigo-200' : 'bg-gray-50 border border-gray-200'}`}>
      <Icon size={14} className={active ? 'text-indigo-600' : 'text-gray-500'} />
      <span className="font-medium text-gray-700">{label}</span>
      {sublabel && <span className="text-gray-400 text-[10px]">{sublabel}</span>}
    </div>
  )
}

function Arrow() {
  return <span className="text-gray-300 text-lg">→</span>
}

function CapabilityCard({ emoji, title, description }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm hover:shadow-md transition-shadow">
      <span className="text-2xl">{emoji}</span>
      <h3 className="font-semibold text-gray-900 text-sm mt-2">{title}</h3>
      <p className="text-xs text-gray-500 mt-1 leading-relaxed">{description}</p>
    </div>
  )
}
