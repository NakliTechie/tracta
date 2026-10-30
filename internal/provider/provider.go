// Package provider sends tracta's questions to a model endpoint.
package provider

import (
	"bytes"
	"context"
	"encoding/json"

	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Client answers one prompt. topLogprobs > 0 also asks for the first token's top log-probabilities,
// where the endpoint supports them (nil otherwise).
type Client interface {
	Chat(ctx context.Context, prompt string, maxTokens, topLogprobs int) (string, map[string]float64, error)
}

// Options are the parts of a watch that shape requests.
type Options struct {
	API, Endpoint, Model, KeyEnv string
	ExtraBody                    map[string]any
	RPS                          float64
}

// New builds the client for an API kind.
func New(o Options) (Client, error) {
	key := ""
	if o.KeyEnv != "" {
		key = os.Getenv(o.KeyEnv)
		if key == "" {
			return nil, fmt.Errorf("environment variable %s is not set", o.KeyEnv)
		}
	}
	pace := &pacer{}
	if o.RPS > 0 {
		pace.gap = time.Duration(float64(time.Second) / o.RPS)
	}
	switch o.API {
	case "openai", "":
		return &openAI{url: strings.TrimRight(o.Endpoint, "/") + "/chat/completions", model: o.Model, key: key,
			extra: o.ExtraBody, pace: pace, http: &http.Client{Timeout: 120 * time.Second}}, nil
	case "anthropic":
		return newAnthropic(o, key, pace), nil
	}
	return nil, fmt.Errorf("unknown api %q (want openai or anthropic)", o.API)
}

type pacer struct {
	gap  time.Duration
	last time.Time
}

func (p *pacer) wait() {
	if d := time.Until(p.last.Add(p.gap)); d > 0 {
		time.Sleep(d)
	}
	p.last = time.Now()
}

// openAI talks to any OpenAI-compatible /chat/completions endpoint with the standard library.
type openAI struct {
	url, model, key string
	extra           map[string]any
	pace            *pacer
	http            *http.Client
}

func (c *openAI) Chat(ctx context.Context, prompt string, maxTokens, topLogprobs int) (string, map[string]float64, error) {
	body := map[string]any{}
	for k, v := range c.extra {
		body[k] = v
	}
	body["model"], body["temperature"], body["max_tokens"] = c.model, 0, maxTokens
	body["messages"] = []map[string]string{{"role": "user", "content": prompt}}
	if topLogprobs > 0 {
		body["logprobs"], body["top_logprobs"] = true, topLogprobs
	}
	data, _ := json.Marshal(body)
	var out struct {
		Choices []struct {
			Message  struct{ Content string } `json:"message"`
			Logprobs *struct {
				Content []struct {
					TopLogprobs []struct {
						Token   string  `json:"token"`
						Logprob float64 `json:"logprob"`
					} `json:"top_logprobs"`
				} `json:"content"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	var lastErr error
	for attempt := range 4 {
		if attempt > 0 {
			time.Sleep(time.Duration(5<<attempt) * time.Second)
		}
		c.pace.wait()
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")
		if c.key != "" {
			req.Header.Set("Authorization", "Bearer "+c.key)
		}
		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("cannot reach %s: %w", c.url, err)
			continue
		}
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		switch {
		case resp.StatusCode == 429 || resp.StatusCode >= 500:
			lastErr = fmt.Errorf("HTTP %d from %s: %.300s", resp.StatusCode, c.url, raw)
			continue
		case resp.StatusCode != 200:
			return "", nil, fmt.Errorf("HTTP %d from %s: %.300s", resp.StatusCode, c.url, raw)
		}
		if err := json.Unmarshal(raw, &out); err != nil || len(out.Choices) == 0 {
			return "", nil, fmt.Errorf("unexpected response from %s: %.300s", c.url, raw)
		}
		ch := out.Choices[0]
		var top map[string]float64
		if topLogprobs > 0 && ch.Logprobs != nil && len(ch.Logprobs.Content) > 0 {
			top = map[string]float64{}
			for _, t := range ch.Logprobs.Content[0].TopLogprobs {
				top[t.Token] = t.Logprob
			}
		}
		return ch.Message.Content, top, nil
	}
	return "", nil, lastErr
}
