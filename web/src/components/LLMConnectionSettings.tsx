import { Check, Eye, EyeOff, KeyRound, LoaderCircle, Save, Server, TriangleAlert } from "lucide-react";
import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  getLLMSettings,
  saveLLMSettings,
  type BackendStatus,
  type LLMSettings,
  type LLMSettingsUpdate,
} from "../api/client";

export type ChatApiMode = "本地 (LM Studio)" | "线上 API";

type LLMConnectionSettingsProps = {
  apiMode: ChatApiMode;
  backendStatus: BackendStatus;
  flash: (message: string) => void;
};

type FormState = {
  localApiBase: string;
  localModel: string;
  localApiKey: string;
  onlineApiBase: string;
  onlineModel: string;
  onlineApiKey: string;
};

const emptyForm: FormState = {
  localApiBase: "http://127.0.0.1:1234/v1",
  localModel: "local-model",
  localApiKey: "",
  onlineApiBase: "",
  onlineModel: "deepseek-v4-flash",
  onlineApiKey: "",
};

function persistenceLabel(settings: LLMSettings | null) {
  if (!settings) return ".env";
  if (settings.persistence.runtimeMode === "portable") return "portable-settings.env";
  if (settings.persistence.runtimeMode === "docker") return ".env（宿主项目目录）";
  return settings.persistence.fileName || ".env";
}

export function LLMConnectionSettings({ apiMode, backendStatus, flash }: LLMConnectionSettingsProps) {
  const [settings, setSettings] = useState<LLMSettings | null>(null);
  const [form, setForm] = useState<FormState>(emptyForm);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [clearLocalKey, setClearLocalKey] = useState(false);
  const [clearOnlineKey, setClearOnlineKey] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [reloadToken, setReloadToken] = useState(0);
  const firstInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (backendStatus !== "online") {
      setSettings(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void getLLMSettings(controller.signal)
      .then((payload) => {
        setSettings(payload);
        setForm({
          localApiBase: payload.local.apiBase,
          localModel: payload.local.model,
          localApiKey: "",
          onlineApiBase: payload.online.apiBase,
          onlineModel: payload.online.model,
          onlineApiKey: "",
        });
      })
      .catch((loadError: Error) => {
        if (loadError.name !== "AbortError") setError(`配置读取失败：${loadError.message}`);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [backendStatus, reloadToken]);

  const local = apiMode === "本地 (LM Studio)";
  const keyConfigured = local ? settings?.local.apiKeyConfigured : settings?.online.apiKeyConfigured;
  const clearKey = local ? clearLocalKey : clearOnlineKey;
  const apiBase = local ? form.localApiBase : form.onlineApiBase;
  const model = local ? form.localModel : form.onlineModel;
  const apiKey = local ? form.localApiKey : form.onlineApiKey;
  const prefix = local ? "local" : "online";

  const patchActive = (field: "apiBase" | "model" | "apiKey", value: string) => {
    const key = `${prefix}${field[0].toUpperCase()}${field.slice(1)}` as keyof FormState;
    setForm((current) => ({ ...current, [key]: value }));
    if (field === "apiKey" && value) {
      if (local) setClearLocalKey(false);
      else setClearOnlineKey(false);
    }
    setMessage("");
    setError("");
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (backendStatus !== "online" || saving || !settings?.persistence.writable) return;
    setSaving(true);
    setMessage("");
    setError("");
    const payload: LLMSettingsUpdate = {
      localApiBase: form.localApiBase,
      localModel: form.localModel,
      clearLocalApiKey: clearLocalKey,
      onlineApiBase: form.onlineApiBase,
      onlineModel: form.onlineModel,
      clearOnlineApiKey: clearOnlineKey,
    };
    if (form.localApiKey) payload.localApiKey = form.localApiKey;
    if (form.onlineApiKey) payload.onlineApiKey = form.onlineApiKey;
    try {
      const saved = await saveLLMSettings(payload);
      setSettings(saved);
      setForm((current) => ({ ...current, localApiKey: "", onlineApiKey: "" }));
      setClearLocalKey(false);
      setClearOnlineKey(false);
      setShowKey(false);
      const savedMessage = `已保存到 ${persistenceLabel(saved)}，后续对话立即使用新配置。`;
      setMessage(savedMessage);
      flash("LLM API 配置已保存");
    } catch (saveError) {
      setError(`保存失败：${(saveError as Error).message}`);
      window.requestAnimationFrame(() => firstInputRef.current?.focus());
    } finally {
      setSaving(false);
    }
  };

  const toggleClear = () => {
    if (local) setClearLocalKey((value) => !value);
    else setClearOnlineKey((value) => !value);
    patchActive("apiKey", "");
  };

  if (backendStatus !== "online") {
    return <p className="llm-config-offline"><TriangleAlert size={14} aria-hidden="true" />后端离线，无法读取或保存 API 配置。</p>;
  }

  return (
    <form className="llm-connection-form" onSubmit={submit} aria-busy={loading || saving}>
      <header>
        <div><Server size={15} aria-hidden="true" /><b>上游 API 配置</b></div>
        <span className="mono">{local ? "LOCAL" : "ONLINE"}</span>
      </header>
      <p className="llm-config-target">保存到 <strong>{persistenceLabel(settings)}</strong>；Key 不会从后端明文回传。</p>
      {loading && <p className="llm-config-loading" role="status"><LoaderCircle size={14} aria-hidden="true" />正在读取配置…</p>}

      <label htmlFor={`${prefix}-api-base`}>API URL</label>
      <input
        ref={firstInputRef}
        id={`${prefix}-api-base`}
        type="url"
        required={local}
        value={apiBase}
        disabled={loading || saving}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? "llm-config-error" : undefined}
        onChange={(event) => patchActive("apiBase", event.target.value)}
        placeholder={local ? "http://127.0.0.1:1234/v1" : "https://api.example.com/v1"}
      />
      <small>填写 OpenAI 兼容根地址，后端会追加 <code>/chat/completions</code>。</small>

      <label htmlFor={`${prefix}-model`}>模型名称</label>
      <input
        id={`${prefix}-model`}
        required
        value={model}
        disabled={loading || saving}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? "llm-config-error" : undefined}
        onChange={(event) => patchActive("model", event.target.value)}
        placeholder={local ? "local-model" : "deepseek-v4-flash"}
      />

      <label htmlFor={`${prefix}-api-key`}>API Key</label>
      <div className="llm-secret-input">
        <KeyRound size={14} aria-hidden="true" />
        <input
          id={`${prefix}-api-key`}
          type={showKey ? "text" : "password"}
          autoComplete="new-password"
          value={apiKey}
          disabled={loading || saving || clearKey}
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "llm-config-error" : undefined}
          onChange={(event) => patchActive("apiKey", event.target.value)}
          placeholder={keyConfigured ? "已配置；留空保持不变" : "可留空"}
        />
        <button
          type="button"
          aria-label={showKey ? "隐藏本次输入的 API Key" : "显示本次输入的 API Key"}
          aria-pressed={showKey}
          onClick={() => setShowKey((value) => !value)}
          disabled={loading || saving}
        >
          {showKey ? <EyeOff size={15} aria-hidden="true" /> : <Eye size={15} aria-hidden="true" />}
        </button>
      </div>
      <div className="llm-key-state">
        <span>{clearKey ? "保存后清除已配置 Key" : keyConfigured ? "•••••••• 已配置" : "当前未配置 Key"}</span>
        {keyConfigured && <button type="button" onClick={toggleClear}>{clearKey ? "撤销清除" : "清除 Key"}</button>}
      </div>

      <button className="llm-save-button" type="submit" disabled={loading || saving || !settings?.persistence.writable}>
        <Save size={14} aria-hidden="true" />{saving ? "正在保存…" : `保存本地与线上配置`}
      </button>
      {!loading && settings && !settings.persistence.writable && (
        <p className="llm-config-result llm-config-error" role="alert"><TriangleAlert size={14} aria-hidden="true" />配置文件不可写，请检查文件权限。</p>
      )}
      {message && <p className="llm-config-result llm-config-success" role="status"><Check size={14} aria-hidden="true" />{message}</p>}
      {error && (
        <div id="llm-config-error" className="llm-config-result llm-config-error" role="alert">
          <TriangleAlert size={14} aria-hidden="true" /><span>{error}</span>
          {!settings && <button type="button" onClick={() => setReloadToken((value) => value + 1)}>重新读取</button>}
        </div>
      )}
    </form>
  );
}
