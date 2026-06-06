# Cannot retrieve past agent history after connection lost (Jun 6 1:27 PM)

**Session ID:** `b9621586-6a95-425e-857d-27fa41984cb7`

**Recovered:** 2026-06-06 13:42

**Source:** `/users/cristian1/.cursor/projects/1780741538920/agent-transcripts/b9621586-6a95-425e-857d-27fa41984cb7/b9621586-6a95-425e-857d-27fa41984cb7.jsonl`

---


---

## You (1)

there is an issue on lumi, i cannno retreive past agent history if at some point the  connection is lost


## Assistant

Investigating how Lumi handles agent history and connection recovery after disconnects.


## Assistant

This is a known Cursor Remote SSH issue, and your LUMI logs match it. The history is usually still on disk — the UI loses the link to it after a disconnect.

## What’s going wrong

After a disconnect, Cursor often fails to reattach to the **same workspace session**. Instead it opens a **new** one with a different workspace ID. Chat history is keyed to that ID, so past agents disappear from the sidebar even though the data still exists.

On your account I can see:

| Workspace ID | Has transcripts | Notes |
|---|---|---|
| `1780681844355` | Yes — 4 chats (including a 541 KB Team3 session) | Your main project at `/scratch/project_462001520/Team3/Team3` |
| `1780741538920` | Only this current chat | New session after reconnect |
| `tmp-*` folders | No transcripts | Transient sessions with no folder bound |

Your logs also show the current session started with **no folder open**:

```text
workspacePathCount: 0, workspacePaths: []
```

That means history won’t show in the UI until you reconnect to the **exact same project folder**, not just “Connect to LUMI” again.

There were also several disconnects today (13:00, 13:03, 13:06, 13:25), which fits the pattern.

## Your history is not lost

Transcripts are still on LUMI at:

```bash
~/.cursor/projects/1780681844355/agent-transcripts/
```

That includes your large Team3 agent session (`f9492c25-...`, ~541 KB). The UI just isn’t pointing at workspace `1780681844355` right now.

## How to get history back in Cursor

1. **Reconnect to the saved project**, not a fresh SSH host connection  
   - Open the **same** remote folder: `/scratch/project_462001520/Team3/Team3`  
   - Use **File → Open Recent** or the saved Remote SSH workspace entry if you have one  
   - Avoid starting a blank “Connect to Host” session without opening the folder

2. **Use an SSH config alias on your local machine** (Cursor forum workaround for the Remote SSH reconnect regression):

```sshconfig
Host lumi
    HostName lumi.csc.fi
    User cristian1
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
```

Then connect via **Remote SSH → lumi** (the alias), not `user@host`.

3. **If reconnect loops or hangs**, on LUMI run:

```bash
pkill -f multiplex-server
```

Then in Cursor: **Developer: Reload Window** (`Cmd/Ctrl+Shift+P`).

4. **If the sidebar is still empty** but you’re in the right folder, the chat index on your **local** machine may have lost the workspace binding (`workspace: null` in `state.vscdb`). That’s a separate Cursor bug; the [cursor-chat-recovery](https://github.com/cpeoples/cursor-chat-recovery) tool can rebind it. Quit Cursor fully before running it.

## Read history directly from LUMI (workaround)

If the UI still won’t show old chats, you can read the raw transcripts:

```bash
ls ~/.cursor/projects/1780681844355/agent-transcripts/*/
# Each folder has a .jsonl file with the full conversation
less ~/.cursor/projects/1780681844355/agent-transcripts/f9492c25-0a1d-4b70-9f3b-f2363ca3de8b/f9492c25-0a1d-4b70-9f3b-f2363ca3de8b.jsonl
```

## Prevention on LUMI

- **Always open the project folder** before chatting — don’t use agent chat in a host-only session with no folder.
- **Keep the path stable** — don’t rename or move the project directory (that changes the workspace ID).
- **Use SSH keepalive** (config above) to reduce idle disconnects on HPC login nodes.
- **Update Cursor** — recent Remote SSH regressions around saved-project reconnect are being fixed upstream.

---

If you want, I can put together a small script on LUMI that lists your agent transcripts by date/topic and prints summaries, so you always have a fallback when the UI drops history after a disconnect.
