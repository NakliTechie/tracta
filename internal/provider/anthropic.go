package provider

import (
	"context"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicClient talks to the Claude Messages API through the official Go SDK. The API takes no
// sampling parameters, so answers are not greedy; tracta measures that noise from the baselines.
// Thinking is set per watch through extra_body (for example {"thinking": {"type": "disabled"}} on
// Claude Haiku 5.5, or {"type": "between_tools"} on Claude Sonnet 5.5), so short answers are not
// spent on thinking tokens. Log-probabilities are not offered, so the fingerprint is skipped.
type anthropicClient struct {
	client anthropic.Client
	model  string
	opts   []option.RequestOption
	pace   *pacer
}

func newAnthropic(o Options, key string, pace *pacer) *anthropicClient {
	clientOpts := []option.RequestOption{option.WithMaxRetries(4)}
	if key != "" {
		clientOpts = append(clientOpts, option.WithAPIKey(key))
	}
	if o.Endpoint != "" && !strings.Contains(o.Endpoint, "api.anthropic.com") {
		clientOpts = append(clientOpts, option.WithBaseURL(o.Endpoint))
	}
	var reqOpts []option.RequestOption
	for k, v := range o.ExtraBody {
		reqOpts = append(reqOpts, option.WithJSONSet(k, v))
	}
	return &anthropicClient{client: anthropic.NewClient(clientOpts...), model: o.Model, opts: reqOpts, pace: pace}
}

func (c *anthropicClient) Chat(ctx context.Context, prompt string, maxTokens, topLogprobs int) (string, map[string]float64, error) {
	c.pace.wait()
	resp, err := c.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: int64(maxTokens),
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(prompt))},
	}, c.opts...)
	if err != nil {
		return "", nil, err
	}
	var text strings.Builder
	for _, block := range resp.Content {
		if b, ok := block.AsAny().(anthropic.TextBlock); ok {
			text.WriteString(b.Text)
		}
	}
	return text.String(), nil, nil
}
