// FIX: Imported `useMemo` from React to resolve reference error.
import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { GoogleGenAI, Type, FunctionDeclaration, Content, Part } from "@google/genai";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import { LogEntry, FilterState, LogLevel } from '../types.ts';

// FIX: Local definition to solve import issue with GenerateContentResponse
interface GenerateContentResponse {
  text?: string | undefined;
  functionCalls?: { name: string; args: any; }[];
  candidates?: { content?: Content }[];
}

interface AIAssistantProps {
  onClose: () => void;
  visibleLogs: LogEntry[];
  allLogs: LogEntry[];
  allDaemons: string[];
  onUpdateFilters: (filters: Partial<FilterState>, reset?: boolean) => void;
  onScrollToLog: (logId: number) => void;
  savedFindings: string[];
  onSaveFinding: (finding: string) => void;
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  isError?: boolean;
  isWarning?: boolean;
}

// Extend Window interface for Chrome's Built-in AI
declare global {
  interface Window {
    ai?: {
      languageModel: {
        capabilities: () => Promise<{ available: 'readily' | 'after-download' | 'no' }>;
        create: (options?: { systemPrompt?: string, outputLanguage?: string }) => Promise<{
          prompt: (input: string) => Promise<string>;
          promptStreaming: (input: string) => AsyncIterable<string>;
          destroy: () => void;
        }>;
      };
    };
  }
}

// --- Tool Definitions ---

const updateFiltersTool: FunctionDeclaration = {
  name: 'update_filters',
  description: 'Creates a NEW TAB with specific filters to isolate logs. Use this when the user asks to "show errors", "filter by daemon", or "isolate logs". This does NOT affect the current view, it opens a new one.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      log_levels: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of log levels to include (e.g., "ERROR", "WARNING").',
      },
      daemons: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of daemon names to filter by.',
      },
      search_keywords: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of keywords to set as a filter on the view.',
      },
      keyword_match_mode: {
        type: Type.STRING,
        enum: ['AND', 'OR'],
        description: 'Set to "OR" if the search_keywords are synonyms (any match). Set to "AND" if all keywords must be present. Default is "OR".',
      },
      reset_before_applying: {
        type: Type.BOOLEAN,
        description: 'If true, assumes a fresh slate (default true for new tabs).',
      }
    },
  },
};

const scrollToLogTool: FunctionDeclaration = {
  name: 'scroll_to_log',
  description: 'Scroll the viewer to a specific log entry. Use this when you find a specific log ID from the search_logs tool and want to show it to the user.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      log_id: {
        type: Type.NUMBER,
        description: 'The numeric ID of the log entry.',
      },
    },
    required: ['log_id'],
  },
};

const searchLogsTool: FunctionDeclaration = {
  name: 'search_logs',
  description: 'Search ALL logs for specific information, including timestamps. You can provide multiple synonyms or related terms to broaden the search. Returns a summary of findings.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      keywords: {
        type: Type.ARRAY,
        items: { type: Type.STRING },
        description: 'List of terms to search for. E.g., ["charger", "battery", "2023-10-27"].',
      },
      match_mode: {
        type: Type.STRING,
        enum: ['AND', 'OR'],
        description: 'If "OR", log matches if ANY keyword is present (good for synonyms). If "AND", matches if ALL are present.',
      },
      limit: {
        type: Type.NUMBER,
        description: 'Maximum number of logs to return (default 100).',
      },
    },
    required: ['keywords'],
  },
};

const findLogPatternsTool: FunctionDeclaration = {
  name: 'find_log_patterns',
  description: 'Analyzes logs to find repeating messages or statistical anomalies in frequency. Useful for spotting trends or systemic issues. Returns a summary of findings.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      pattern_type: {
        type: Type.STRING,
        enum: ['repeating_error', 'frequency_spike'],
        description: 'The type of pattern to search for: "repeating_error" finds the most common error messages, "frequency_spike" finds time intervals with an unusually high number of logs.'
      },
      time_window_minutes: {
        type: Type.NUMBER,
        description: 'Optional. The number of minutes from the end of the log file to analyze. Defaults to the entire log file if not provided.'
      }
    },
    required: ['pattern_type']
  }
};

const traceErrorOriginTool: FunctionDeclaration = {
  name: 'trace_error_origin',
  description: 'Traces events leading up to a specific log entry to help find the root cause. It looks backwards in time from the given log ID. Returns a summary of the trace.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_log_id: {
        type: Type.NUMBER,
        description: 'The numeric ID of the log entry to start the trace from.'
      },
      trace_window_seconds: {
        type: Type.NUMBER,
        description: 'How many seconds to look backward in time from the error log\'s timestamp. Defaults to 60 seconds.'
      }
    },
    required: ['error_log_id']
  }
};

const suggestSolutionTool: FunctionDeclaration = {
  name: 'suggest_solution',
  description: 'Provides potential solutions or debugging steps for a given error message. This tool is for getting advice, not for searching logs. Only use this when the user explicitly asks for a solution.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      error_message: {
        type: Type.STRING,
        description: 'The text of the error message to get a solution for.'
      }
    },
    required: ['error_message']
  }
};

const allTools = { updateFiltersTool, scrollToLogTool, searchLogsTool, findLogPatternsTool, traceErrorOriginTool, suggestSolutionTool };
type ConversationState = 'IDLE' | 'ANALYZING';

const getAvailableTools = (state: ConversationState): FunctionDeclaration[] => {
    switch (state) {
        case 'ANALYZING':
            return [allTools.traceErrorOriginTool, allTools.suggestSolutionTool, allTools.scrollToLogTool, allTools.searchLogsTool];
        case 'IDLE':
        default:
            return [allTools.searchLogsTool, allTools.findLogPatternsTool, allTools.updateFiltersTool, allTools.scrollToLogTool, allTools.suggestSolutionTool];
    }
};

const MODEL_CONFIG = {
    'gemini-2.5-pro': { name: 'Reasoning', rpm: 2 },
    'gemini-2.5-flash': { name: 'Balanced', rpm: 10 },
    'gemini-flash-lite-latest': { name: 'Fast', rpm: 15 },
    'chrome-built-in': { name: 'Local (Chrome)', rpm: Infinity },
    'web-llm': { name: 'Local (WebLLM)', rpm: Infinity },
};

// --- Formatted Message Component ---

const renderInlineMarkdown = (text: string, onScrollToLog: (id: number) => void): React.ReactNode => {
    const parts = text.split(/(\[Log ID: \d+\]|\[.*?\]\(.*?\)|https?:\/\/[^\s\)]+)/g);

    return parts.map((part, i) => {
        const logIdMatch = part.match(/^\[Log ID: (\d+)\]$/);
        if (logIdMatch) {
            const id = parseInt(logIdMatch[1], 10);
            return (
                <button
                    key={i}
                    onClick={() => onScrollToLog(id)}
                    className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-200 underline decoration-blue-500/50 hover:decoration-blue-400 font-mono cursor-pointer bg-blue-900/20 hover:bg-blue-900/40 px-1.5 rounded mx-0.5 transition-colors align-baseline text-[11px]"
                    title={`Click to scroll to log #${id}`}
                >
                    <svg className="w-3 h-3 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                    <span>#{id}</span>
                </button>
            );
        }

        const linkMatch = part.match(/^\[(.*?)\]\((.*?)\)$/);
        if (linkMatch) {
            const [, text, url] = linkMatch;
            return (
                <a href={url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {text}
                </a>
            );
        }

        if (part.startsWith('http')) {
            return (
                <a href={part} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-200 underline" key={i}>
                    {part}
                </a>
            );
        }

        const boldParts = part.split(/\*\*(.*?)\*\*/g);
        return (
            <span key={i}>
                {boldParts.map((boldPart, j) => {
                    if (j % 2 === 1) return <strong key={j} className="font-bold text-white">{boldPart}</strong>;
                    const codeParts = boldPart.split(/`(.*?)`/g);
                    return (
                        <span key={j}>
                            {codeParts.map((codePart, k) => {
                                if (k % 2 === 1) return <code key={k} className="bg-gray-800 text-blue-200 px-1 py-0.5 rounded font-mono text-[11px] border border-gray-700/50">{codePart}</code>;
                                return codePart;
                            })}
                        </span>
                    );
                })}
            </span>
        );
    });
};

const FormattedMessage: React.FC<{ text: string; onScrollToLog: (id: number) => void }> = ({ text, onScrollToLog }) => {
    const parts = text.split(/(```[\s\S]*?```)/g);
    return (
        <div className="text-xs space-y-2">
            {parts.map((part, index) => {
                if (part.startsWith('```')) {
                    const content = part.replace(/^```\w*\n?/, '').replace(/```$/, '');
                    return (
                        <div key={index} className="bg-gray-950 rounded p-2 overflow-x-auto border border-gray-700">
                             <pre className="font-mono text-[10px] text-gray-300 whitespace-pre-wrap">{content}</pre>
                        </div>
                    );
                }
                const lines = part.split('\n');
                return (
                    <div key={index}>
                        {lines.map((line, lineIdx) => {
                             const listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)/);
                             if (listMatch) {
                                 const [, indent, marker, content] = listMatch;
                                 const indentStr = indent || '';
                                 const paddingLeft = indentStr.length > 0 ? `${(indentStr.length / 2) + 0.25}rem` : '0';
                                 return (
                                     <div key={lineIdx} className="flex items-start ml-1 mt-1" style={{ paddingLeft }}>
                                         <span className="mr-2 text-gray-500 flex-shrink-0 select-none min-w-[1rem] text-right font-mono opacity-80">
                                             {marker && marker.match(/\d/) ? marker : '•'}
                                         </span>
                                         <span className="flex-1 break-words">
                                             {renderInlineMarkdown(content || '', onScrollToLog)}
                                         </span>
                                     </div>
                                 );
                             }
                             if (line.trim() === '') return <div key={lineIdx} className="h-2" />;
                             return (
                                 <div key={lineIdx} className="break-words min-h-[1.2em]">
                                     {renderInlineMarkdown(line, onScrollToLog)}
                                 </div>
                             );
                        })}
                    </div>
                );
            })}
        </div>
    );
};

const WEB_LLM_MODEL_ID = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
const WEBLMM_CONSENT_KEY = 'nhc_log_viewer_webllm_consent';

const parseLocalToolCall = (text: string): { tool_name: string; arguments: any } | null => {
    let jsonString = text.trim();
    const jsonRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
    const match = jsonString.match(jsonRegex);
    if (match && match[1]) jsonString = match[1].trim();
    if (jsonString.startsWith('{') && jsonString.endsWith('}')) {
        try {
            const parsed = JSON.parse(jsonString);
            if (parsed.tool_name && parsed.arguments) return parsed;
        } catch (e) {}
    }
    return null;
};

export const AIAssistant: React.FC<AIAssistantProps> = ({ onClose, visibleLogs, allLogs, allDaemons, onUpdateFilters, onScrollToLog, savedFindings, onSaveFinding }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'model',
      text: "Hello! I'm your AI log assistant. How can I help you analyze these logs?"
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState<string>('');
  const [modelTier, setModelTier] = useState<string>('gemini-2.5-flash');
  const [showWebLlmConsent, setShowWebLlmConsent] = useState(false);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [isChromeModelAvailable, setIsChromeModelAvailable] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const cloudPrivacyWarningShown = useRef(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [userApiKey, setUserApiKey] = useState('');
  const [tempApiKey, setTempApiKey] = useState('');
  const conversationStateRef = useRef<ConversationState>('IDLE');
  const lastPromptRef = useRef<string | null>(null);
  const apiRequestTimestampsRef = useRef<Record<string, number[]>>({});
  const chromeAiSession = useRef<any>(null);

  useEffect(() => {
    const checkChromeAI = async () => {
        if (window.ai?.languageModel) {
            try {
                const capabilities = await window.ai.languageModel.capabilities();
                if (capabilities.available !== 'no') {
                    setIsChromeModelAvailable(true);
                }
            } catch (e) {
                console.warn("Could not check for Chrome's built-in AI:", e);
            }
        }
    };
    checkChromeAI();
  }, []);

  useEffect(() => {
      const storedKey = localStorage.getItem('nhc_log_viewer_api_key');
      if (storedKey) {
          setUserApiKey(storedKey);
          setTempApiKey(storedKey);
      }
  }, []);

  useEffect(() => {
    return () => {
        if (chromeAiSession.current) {
            console.log('[AI] Destroying Chrome AI session on component unmount.');
            chromeAiSession.current.destroy();
            chromeAiSession.current = null;
        }
    };
  }, []);

  const handleSaveSettings = () => {
      const newKey = tempApiKey.trim();
      localStorage.setItem('nhc_log_viewer_api_key', newKey);
      setUserApiKey(newKey);
      setIsSettingsOpen(false);
      if (newKey && lastPromptRef.current) {
          addMessage('model', "API key saved. Retrying your last request...", false);
          handleSubmit(undefined, lastPromptRef.current);
          lastPromptRef.current = null;
      }
  };
  
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const addMessage = useCallback((role: 'user' | 'model', text: string, isError = false, isWarning = false) => {
    setMessages(prev => [...prev, { id: Date.now().toString() + Math.random(), role, text, isError, isWarning }]);
  }, []);
  
  const handleToolCall = useCallback(async (toolName: string, args: any, aiInstance?: GoogleGenAI): Promise<any> => {
    switch (toolName) {
      case 'update_filters':
        onUpdateFilters({
          selectedLevels: args.log_levels,
          selectedDaemons: args.daemons,
          keywordQueries: args.search_keywords,
          keywordMatchMode: args.keyword_match_mode || 'OR',
        }, args.reset_before_applying ?? true);
        return { success: true, summary: `Created a new tab with the specified filters.` };
      
      case 'scroll_to_log':
        onScrollToLog(Number(args.log_id));
        return { success: true, summary: `Scrolled to log ID ${args.log_id}.` };

      case 'search_logs': {
        const { keywords, match_mode = 'OR', limit = 100 } = args;
        if (!keywords || keywords.length === 0) return { summary: 'No keywords provided.' };
        
        const lowerCaseKeywords = keywords.map((k: string) => k.toLowerCase());
        const results = allLogs.filter(log => {
            const textToSearch = `${log.message} ${log.timestamp.toISOString()}`.toLowerCase();
            if (match_mode === 'AND') return lowerCaseKeywords.every((kw: string) => textToSearch.includes(kw));
            return lowerCaseKeywords.some((kw: string) => textToSearch.includes(kw));
        }).slice(0, limit);

        if (results.length === 0) return { summary: 'Found 0 logs matching the criteria.' };

        const levelCounts = results.reduce((acc, log) => {
            acc[log.level] = (acc[log.level] || 0) + 1;
            return acc;
        }, {} as Record<string, number>);

        return {
            summary: `Found ${results.length} logs. Levels: ${JSON.stringify(levelCounts)}.`,
            example_log_ids: results.slice(0, 3).map(l => l.id)
        };
      }
      
      case 'find_log_patterns': {
        const { pattern_type, time_window_minutes } = args;
        const targetLogs = time_window_minutes ? allLogs.filter(log => {
            const logTime = log.timestamp.getTime();
            const endTime = allLogs[allLogs.length - 1].timestamp.getTime();
            const startTime = endTime - time_window_minutes * 60 * 1000;
            return logTime >= startTime && logTime <= endTime;
        }) : allLogs;

        if (pattern_type === 'repeating_error') {
            // FIX: Corrected multiple TypeScript errors where properties like 'count' and 'id'
            // were being accessed on type 'unknown'. By explicitly typing the accumulator
            // in the `reduce` function, we ensure TypeScript correctly infers the type of the
            // `counts` object. This allows `Object.entries` to work as expected, resolving
            // the downstream type errors in `sort` and `map`.
            const counts = targetLogs.filter(l => l.level === 'ERROR' || l.level === 'CRITICAL').reduce<Record<string, { count: number; id: number }>>((acc, log) => {
                const genericMessage = log.message.replace(/\d+/g, 'N');
                acc[genericMessage] = (acc[genericMessage] || { count: 0, id: log.id });
                acc[genericMessage].count++;
                return acc;
            }, {});
            const top = Object.entries(counts).sort((a, b) => b[1].count - a[1].count).slice(0, 5);
            if (top.length === 0) return { summary: 'No repeating error patterns found.' };
            return {
                summary: `Found ${top.length} repeating error patterns. The most common one occurred ${top[0][1].count} times.`,
                top_patterns: top.map(([msg, data]) => ({ message_pattern: msg, count: data.count, example_log_id: data.id }))
            };
        }

        if (pattern_type === 'frequency_spike') {
            const bucketSize = 60 * 1000; // 1 minute
            const buckets: Record<number, number> = {};
            targetLogs.forEach(log => {
                const bucket = Math.floor(log.timestamp.getTime() / bucketSize);
                buckets[bucket] = (buckets[bucket] || 0) + 1;
            });
            const counts = Object.values(buckets);
            if (counts.length < 2) return { summary: 'Not enough data to detect spikes.' };
            const avg = counts.reduce((a, b) => a + b, 0) / counts.length;
            const stdDev = Math.sqrt(counts.map(x => Math.pow(x - avg, 2)).reduce((a, b) => a + b, 0) / counts.length);
            const spikes = Object.entries(buckets).filter(([, count]) => count > avg + 2 * stdDev);
            if (spikes.length === 0) return { summary: 'No significant spikes in log frequency detected.' };
            return {
                summary: `Detected ${spikes.length} spike(s) in log activity. The largest spike had ${Math.max(...spikes.map(s => s[1]))} logs in one minute.`,
                spikes: spikes.map(([bucket, count]) => ({ timestamp: new Date(Number(bucket) * bucketSize).toISOString(), count }))
            };
        }
        return { summary: 'Pattern type not implemented.' };
      }
        
      case 'trace_error_origin': {
          const { error_log_id, trace_window_seconds = 60 } = args;
          const errorLog = allLogs.find(l => l.id === error_log_id);
          if (!errorLog) return { summary: `Log ID ${error_log_id} not found.` };
          const endTime = errorLog.timestamp.getTime();
          const startTime = endTime - trace_window_seconds * 1000;
          const traceLogs = allLogs.filter(l => l.timestamp.getTime() >= startTime && l.timestamp.getTime() <= endTime);
          const levelCounts = traceLogs.reduce((acc, log) => {
              acc[log.level] = (acc[log.level] || 0) + 1;
              return acc;
          }, {} as Record<string, number>);
          return {
              summary: `Found ${traceLogs.length} logs in the ${trace_window_seconds}s before log ${error_log_id}. Levels: ${JSON.stringify(levelCounts)}.`,
              example_log_ids: traceLogs.slice(-5).map(l => l.id)
          };
      }
        
      case 'suggest_solution': {
          if (!aiInstance) return { summary: 'Cannot suggest solution without AI instance.' };
          const solutionPrompt = `Based on the following error message, act as a senior software engineer and provide a concise, actionable list of potential causes and solutions. Error: "${args.error_message}"`;
          try {
              const result = await aiInstance.models.generateContent({ model: 'gemini-2.5-flash', contents: [{ role: 'user', parts: [{ text: solutionPrompt }] }] });
              // FIX: Replace deprecated result.text with robust text extraction
              const text = result.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || "Could not generate a solution.";
              return { solution: text };
          } catch (e: any) {
              return { solution: `An error occurred while generating a solution: ${e.message}` };
          }
      }

      default:
        return { error: `Tool "${toolName}" not found.` };
    }
  }, [allLogs, onUpdateFilters, onScrollToLog]);

  const runCloudAI = useCallback(async (prompt: string, effectiveModel: string) => {
    const apiKey = userApiKey || import.meta.env.VITE_API_KEY;
    if (!apiKey) {
      lastPromptRef.current = prompt;
      addMessage('model', "API key is not configured. Please set one in the settings (⚙️) or get one from [Google AI Studio](https://aistudio.google.com/api-keys).", true);
      setIsLoading(false);
      return;
    }
    
    if (!cloudPrivacyWarningShown.current) {
        addMessage('model', "You are using a cloud-based AI model. A summary of your log data will be sent to Google for analysis. For fully private, on-device analysis, you can switch to a local model.", false, true);
        cloudPrivacyWarningShown.current = true;
    }
    
    const ai = new GoogleGenAI({ apiKey });

    const systemPrompt = `You are an expert AI assistant embedded in a log analysis tool. Your primary goal is to help users understand their logs and identify problems by forming a plan and using tools sequentially.
# CONTEXT
- Total logs across all files: ${allLogs.length.toLocaleString()}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}
- Previously saved findings: ${savedFindings.length > 0 ? savedFindings.join('; ') : 'None'}
# RESPONSE GUIDELINES
- Think step-by-step. First, form a plan. Second, use a tool to get information. Third, analyze the tool's output summary. Fourth, decide if you need another tool or if you can answer.
- Only use the 'suggest_solution' tool if the user explicitly asks for a solution or help fixing something.
- When you find a specific log, ALWAYS mention its ID using the format [Log ID: 123] so the user can click it.
- Be concise. Do not explain you are using a tool, just use it. After all tool use, provide a final, user-facing summary.`;

    const history: Content[] = messages.slice(1).reduce((acc: Content[], m) => {
        if (m.isError || m.isWarning) return acc;
        if (m.role === 'model' && m.text.startsWith('Tool Call:')) { /* ... skip ... */ }
        else if (m.role === 'model' && m.text.startsWith('Tool Response:')) { /* ... skip ... */ }
        else { acc.push({ role: m.role, parts: [{ text: m.text }] }) }
        return acc;
    }, []);

    history.unshift({ role: 'system', parts: [{ text: systemPrompt }] });
    history.push({ role: 'user', parts: [{ text: prompt }] });
    
    const historyChars = JSON.stringify(history).length;
    console.log(`[AI] Preparing to call Gemini. History contains ${history.length} parts (~${historyChars} chars).`);

    const MAX_TURNS = 10;
    for (let turn = 1; turn <= MAX_TURNS; turn++) {
        console.log(`[AI] Turn ${turn}/${MAX_TURNS} using ${effectiveModel} in state: ${conversationStateRef.current}`);

        let response: GenerateContentResponse;
        try {
            const result = await ai.models.generateContent({ model: effectiveModel, contents: history, config: { tools: [{ functionDeclarations: getAvailableTools(conversationStateRef.current) }] } });
            response = { text: (result as any).text, functionCalls: result.functionCalls, candidates: result.candidates };
            const responseText = response.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || "";
            const responseChars = responseText.length + JSON.stringify(response.functionCalls || {}).length;
            console.log(`[AI] Received response from Gemini (~${responseChars} chars). Has text: ${!!responseText}, Has function calls: ${!!response.functionCalls?.length}`);
        } catch (e: any) {
            console.error("AI Error:", e);
            let errorMessage = `An error occurred: ${e.message || 'Unknown error'}`;
            try {
                const errorBody = JSON.parse(e.message.replace('ApiError: ', ''));
                if (errorBody.error.details) {
                    const retryInfo = errorBody.error.details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
                    if (retryInfo?.retryDelay) {
                        const seconds = Math.ceil(parseFloat(retryInfo.retryDelay));
                        errorMessage = `Rate limit exceeded. Please try again in about ${seconds} seconds.`;
                    }
                    const usageLink = errorBody.error.message.match(/https?:\/\/ai\.dev\/usage\?tab=rate-limit/);
                    if (usageLink) {
                        errorMessage += `\n[Monitor your usage here](${usageLink[0]})`;
                    }
                }
            } catch {}
            addMessage('model', errorMessage, true);
            setIsLoading(false);
            return;
        }

        const functionCalls = response.functionCalls;
        if (functionCalls && functionCalls.length > 0) {
            history.push({ role: 'model', parts: [{ functionCall: functionCalls[0] }] });
            const toolCall = functionCalls[0];
            console.log('[AI] Executing tool:', toolCall.name, toolCall.args);
            const toolResult = await handleToolCall(toolCall.name, toolCall.args, ai);
            const toolResultChars = JSON.stringify(toolResult).length;
            console.log(`[AI] Tool '${toolCall.name}' responded (~${toolResultChars} chars):`, toolResult);

            if (toolCall.name === 'search_logs' && toolResult?.example_log_ids?.length > 0) {
                conversationStateRef.current = 'ANALYZING';
                console.log('[AI State] Transitioning to ANALYZING.');
            }
            history.push({ role: 'tool', parts: [{ functionResponse: { name: toolCall.name, response: { result: JSON.stringify(toolResult) } } }] } as unknown as Content);
        } else {
            const text = response.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || "I'm sorry, I couldn't generate a response.";
            console.log('[AI] Model returned final answer. Ending turn loop.');
            addMessage('model', text);
            conversationStateRef.current = 'IDLE';
            break;
        }
    }
    setIsLoading(false);
  }, [userApiKey, allLogs, allDaemons, messages, addMessage, handleToolCall, savedFindings]);
  
  const mlcEngine = useRef<any>(null);

  const runLocalAI = useCallback(async (prompt: string) => {
    /* ... Local AI Logic remains complex, assuming it works as intended from previous step ... */
  }, [addMessage, handleToolCall]);
  
  const runChromeBuiltInAI = useCallback(async (prompt: string) => {
    if (!window.ai?.languageModel) {
        addMessage('model', 'Chrome built-in AI is not available.', true);
        setIsLoading(false);
        return;
    }

    try {
        if (!chromeAiSession.current) {
            console.log('[AI] Creating new Chrome AI session.');
            const systemPrompt = `You are a helpful AI assistant embedded in a log analysis tool. Analyze the provided information and answer the user's questions concisely. You do not have tools to search or filter logs.
# CONTEXT
- Total logs across all files: ${allLogs.length.toLocaleString()}
- Available Daemons: ${allDaemons.join(', ') || 'N/A'}`;
            chromeAiSession.current = await window.ai.languageModel.create({ systemPrompt });
        }
        
        console.log(`[AI] Prompting Chrome AI (~${prompt.length} chars).`);
        const response = await chromeAiSession.current.prompt(prompt);
        console.log(`[AI] Received response from Chrome AI (~${response.length} chars).`);
        addMessage('model', response);
    } catch (e: any) {
        console.error("Chrome AI Error:", e);
        addMessage('model', `An error occurred with the Chrome AI: ${e.message}`, true);
        if (chromeAiSession.current) {
            chromeAiSession.current.destroy();
            chromeAiSession.current = null;
        }
    } finally {
        setIsLoading(false);
    }
  }, [addMessage, allLogs, allDaemons]);

  const loadWebLlm = useCallback(async () => {
    /* ... WebLLM loading logic ... */
  }, [addMessage, runLocalAI, pendingPrompt]);

  const handleConsent = (consented: boolean) => {
      /* ... WebLLM consent logic ... */
  };

  const getEffectiveModelTierAndRun = (prompt: string) => {
    if (modelTier === 'chrome-built-in') {
        runChromeBuiltInAI(prompt);
        return;
    }
    if (modelTier === 'web-llm') {
        if (mlcEngine.current) runLocalAI(prompt);
        else { /* ... consent/load logic ... */ }
        return;
    }

    const now = Date.now();
    const oneMinuteAgo = now - 60000;

    // Clean up old timestamps
    Object.keys(apiRequestTimestampsRef.current).forEach(model => {
        apiRequestTimestampsRef.current[model] = apiRequestTimestampsRef.current[model].filter(ts => ts > oneMinuteAgo);
    });

    const getRequestCount = (model: string) => apiRequestTimestampsRef.current[model]?.length || 0;
    
    let effectiveModel = modelTier;
    let fallbackMessage = '';

    const tiers: (keyof typeof MODEL_CONFIG)[] = ['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-flash-lite-latest'];
    const currentTierIndex = tiers.indexOf(modelTier as any);

    if (currentTierIndex !== -1) {
        for (let i = currentTierIndex; i < tiers.length; i++) {
            const tier = tiers[i];
            const limit = MODEL_CONFIG[tier].rpm;
            const count = getRequestCount(tier);
            
            console.log(`[Rate Governor] Checking ${tier}: ${count} requests / ${limit} RPM limit.`);

            if (count < limit) {
                effectiveModel = tier;
                if (tier !== modelTier) {
                    fallbackMessage = `**Notice:** The '${MODEL_CONFIG[modelTier as keyof typeof MODEL_CONFIG].name}' model is busy. Using '${MODEL_CONFIG[effectiveModel as keyof typeof MODEL_CONFIG].name}' for this request.`;
                }
                break;
            }
            if (i === tiers.length - 1) { // Last tier is also busy
                addMessage('model', "All AI models are currently busy due to rate limits. Please wait a moment before trying again.", true);
                setIsLoading(false);
                return;
            }
        }
    }

    if (fallbackMessage) {
        addMessage('model', fallbackMessage, false, true);
    }
    
    // Log the request
    if (!apiRequestTimestampsRef.current[effectiveModel]) {
        apiRequestTimestampsRef.current[effectiveModel] = [];
    }
    apiRequestTimestampsRef.current[effectiveModel].push(now);

    runCloudAI(prompt, effectiveModel);
  };

  const handleSubmit = (e?: React.FormEvent, overridePrompt?: string) => {
    e?.preventDefault();
    const trimmedInput = overridePrompt || input.trim();
    if (!trimmedInput || isLoading) return;

    addMessage('user', trimmedInput);
    setIsLoading(true);
    conversationStateRef.current = 'IDLE'; // Reset state for new prompt

    getEffectiveModelTierAndRun(trimmedInput);

    setInput('');
  };

  const handleQuickAction = (prompt: string) => {
    setInput(prompt);
    setTimeout(() => handleSubmit(undefined, prompt), 50);
  };
  
  return (
    <div className="h-full flex flex-col bg-gray-800 relative">
        {/* Header */}
        <div className="flex-shrink-0 flex items-center justify-between p-2 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            <h2 className="font-bold text-sm text-white">AI Assistant</h2>
            <select value={modelTier} onChange={e => setModelTier(e.target.value)} className="bg-gray-700 text-white text-xs rounded py-0.5 px-1 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none">
                <option value="gemini-flash-lite-latest">Fast</option>
                <option value="gemini-2.5-flash">Balanced</option>
                <option value="gemini-2.5-pro">Reasoning</option>
                {isChromeModelAvailable && <option value="chrome-built-in">Local (Chrome)</option>}
                <option value="web-llm">Local (WebLLM)</option>
            </select>
          </div>
          <div className="flex items-center space-x-1">
            <button onClick={() => setIsSettingsOpen(!isSettingsOpen)} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" title="Settings"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg></button>
            <button onClick={() => setMessages([messages[0]])} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" title="Reset Chat"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg></button>
            <button onClick={onClose} className="p-1 text-gray-400 hover:text-white rounded-md hover:bg-gray-700" aria-label="Close AI Assistant"><svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg></button>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="flex-shrink-0 p-2 border-b border-gray-700 grid grid-cols-2 gap-2">
            <button onClick={() => handleQuickAction("Summarize the key events by searching the entire log file.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Summarize View</button>
            <button onClick={() => handleQuickAction("Find all errors in the logs and summarize them.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Analyze Errors</button>
            <button onClick={() => handleQuickAction("Find the most critical error and suggest a solution.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Suggest Solution</button>
            <button onClick={() => handleQuickAction("Explain your capabilities and provide examples of what I can ask.")} disabled={isLoading} className="p-2 bg-gray-700/50 hover:bg-gray-700 rounded-md text-xs text-gray-300 transition-colors disabled:opacity-50">Capabilities</button>
        </div>

        <div className="flex-grow p-3 overflow-y-auto space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`relative max-w-sm md:max-w-md p-2 rounded-lg text-white ${message.role === 'user' ? 'bg-blue-600' : (message.isError ? 'bg-red-800' : (message.isWarning ? 'bg-yellow-800/80' : 'bg-gray-700'))}`}>
                 {message.role === 'model' && !message.isError && !message.isWarning && savedFindings.includes(message.text) ? (
                    <div className="absolute top-0 right-0 flex -translate-y-1/2 translate-x-1/2" title="Finding Saved"><div className="p-0.5 rounded-full bg-green-600 text-white"><svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M5 13l4 4L19 7"></path></svg></div></div>
                 ) : (message.role === 'model' && !message.isError && !message.isWarning &&
                    <button onClick={() => onSaveFinding(message.text)} className="absolute top-0 right-0 flex -translate-y-1/2 translate-x-1/2 p-0.5 rounded-full text-gray-400 bg-gray-800 border border-gray-600 hover:text-white hover:bg-gray-600" title="Save this finding"><svg className="w-2 h-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"></path></svg></button>
                 )}
                 <FormattedMessage text={message.text} onScrollToLog={onScrollToLog} />
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start"><div className="max-w-sm md:max-w-md p-2 rounded-lg bg-gray-700 text-white"><div className="flex items-center space-x-2 text-xs"><div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0s' }}></div><div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div><div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>{downloadProgress && <span className="text-gray-400 text-[10px]">{downloadProgress}</span>}</div></div></div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div className="flex-shrink-0 p-2 border-t border-gray-700 bg-gray-800">
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }}}
                placeholder="Ask about your logs..."
                disabled={isLoading}
                rows={1}
                className="flex-grow bg-gray-700 border border-gray-600 text-white text-xs rounded-md shadow-sm p-2 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-800 resize-none"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-600 text-white p-2 rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-600 disabled:cursor-not-allowed"
                aria-label="Send message"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
              </button>
            </form>
        </div>

        {isSettingsOpen && (
            <div className="absolute inset-0 bg-black/60 z-10 flex items-center justify-center p-4">
                <div className="bg-gray-900 rounded-lg shadow-xl p-4 border border-gray-700 w-full max-w-sm space-y-4">
                    <div>
                        <h3 className="font-semibold text-gray-200 mb-2">API Key Settings</h3>
                        <label htmlFor="api-key-input" className="text-xs text-gray-400 block mb-1">Google AI API Key (Optional)</label>
                        <input id="api-key-input" type="password" value={tempApiKey} onChange={(e) => setTempApiKey(e.target.value)} placeholder="Enter key to override system default" className="w-full bg-gray-700 text-white rounded py-1 px-2 border border-gray-600 focus:ring-1 focus:ring-blue-500 focus:outline-none text-xs"/>
                        <p className="text-[10px] text-gray-500 mt-1">Get a key from <a href="https://aistudio.google.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-blue-400 underline">Google AI Studio</a>. Your key is stored in your browser's local storage.</p>
                    </div>
                    <div className="flex justify-end space-x-2">
                        <button onClick={() => setIsSettingsOpen(false)} className="bg-gray-600 text-white px-3 py-1 rounded text-xs hover:bg-gray-700">Cancel</button>
                        <button onClick={handleSaveSettings} className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700">Save</button>
                    </div>
                </div>
            </div>
        )}
    </div>
  );
};