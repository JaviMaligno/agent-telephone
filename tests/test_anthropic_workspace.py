from agent_telephone.backends.anthropic_api import AnthropicBackend


def test_env_file_sets_key_and_workspace_header(tmp_path):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-test\nANTHROPIC_WORKSPACE_ID=wrkspc_test\n")
    b = AnthropicBackend(guard=None, key_path=env)
    assert b.client.api_key == "sk-test"
    assert b.client.default_headers["anthropic-workspace-id"] == "wrkspc_test"
    assert "sk-test" not in repr(b)
