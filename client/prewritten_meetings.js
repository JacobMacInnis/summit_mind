const samples = [
    // {
    //     label: "Client Meeting Reschedule",
    //     text: `Speaker 1: Can we move the client meeting to Thursday?\nSpeaker 2: I have a call Thursday afternoon.\nSpeaker 3: What about Friday morning?\nSpeaker 2: Friday works.\nSpeaker 1: Let's do Friday.`
    // },
    // {
    //     label: "Sprint Planning Confirmation",
    //     text: `Speaker 1: Morning! Are we still meeting today?\nSpeaker 2: Yes, 2 PM confirmed.\nSpeaker 3: Thanks, see you then!`
    // },
    // {
    //     label: "Login Issue Escalation",
    //     text: `Speaker 1: I'm locked out of my account.\nSpeaker 2: Let me escalate to IT.\nSpeaker 1: Thanks, please email me.`
    // },
    // {
    //     label: "Dark Mode Feature Discussion",
    //     text: `Speaker 1: Should we add dark mode?\nSpeaker 2: Good idea, let's check bandwidth.\nSpeaker 1: I'll draft a proposal.`
    // },
    // {
    //     label: "Conference Attendance Approval",
    //     text: `Speaker 1: Will you attend the conference?\nSpeaker 2: Maybe. I need approval first.\nSpeaker 3: I'll email the manager.`
    // },
    {
        label: "Client Meeting Reschedule",
        text: `Speaker 1: Hey team, the client requested to move our Thursday meeting.
    Speaker 2: I have a quarterly review call on Thursday afternoon.
    Speaker 3: Could we do Friday morning instead? That might work better for everyone.
    Speaker 1: Friday morning sounds good to me. 
    Speaker 2: Same here, but let's avoid 9 AM because I have another meeting.
    Speaker 3: How about 10:30 AM? 
    Speaker 1: Perfect. I'll send out a new invite and let the client know.`
    },
    {
        label: "Sprint Planning Confirmation",
        text: `Speaker 1: Morning everyone! Quick check — are we still meeting today for sprint planning?
    Speaker 2: Yes, but I heard the PM might need to reschedule?
    Speaker 3: They sent an email last night confirming 2 PM.
    Speaker 1: Oh, I missed that email. Thanks for confirming.
    Speaker 4: Should we prepare anything specific beforehand?
    Speaker 2: I suggest reviewing the backlog grooming notes and identifying potential blockers.
    Speaker 1: Good idea. See you all at 2 PM then.`
    },
    {
        label: "Login Issue Escalation",
        text: `Speaker 1: Hi, I’m unable to log into my company account since this morning. 
    Speaker 2: Are you seeing an error message or is it just not responding?
    Speaker 1: It says "incorrect password", but I haven't changed it.
    Speaker 3: Have you tried resetting the password yet?
    Speaker 1: I did, but the reset link expired before I could use it.
    Speaker 2: Alright, I’ll escalate this to the IT support team immediately.
    Speaker 3: I'll open a high-priority ticket as well so they address it within the hour.
    Speaker 1: Thanks, really appreciate the quick help.`
    },
    {
        label: "Dark Mode Discussion",
        text: `Speaker 1: Several customers have requested dark mode. Should we prioritize it for the next sprint?
    Speaker 2: I think it’s a good idea. It could improve accessibility and user experience.
    Speaker 3: Do we have enough UI/UX resources available to design it properly?
    Speaker 4: I can allocate time next week to mockup initial designs.
    Speaker 2: Technically, adding a dark mode toggle is straightforward, but the testing will take time.
    Speaker 1: Ok, let's add a spike story to estimate effort and schedule it accordingly.
    Speaker 3: Agreed. Better to do it right than rush it.`
    },
    {
        label: "Conference Attendance",
        text: `Speaker 1: There's a major AI conference next month. Are we sending representatives this year?
    Speaker 2: I'd love to attend, but I need manager approval first.
    Speaker 3: Same here. I think it would be valuable especially given our new product launch.
    Speaker 4: What’s the expected cost per person?
    Speaker 1: Roughly $1500 including flights and hotel for three days.
    Speaker 3: It would be great exposure for our company.
    Speaker 2: Should we draft a proposal outlining the benefits and expected outcomes?
    Speaker 4: Good idea. Let’s also suggest a short presentation at our all-hands after the event.
    Speaker 1: Perfect. I’ll prepare a proposal by tomorrow.`
    }
];

window.onload = function () {
    const samplesDiv = document.getElementById('samples');
    samples.forEach((sample, idx) => {
        const btn = document.createElement('button');
        btn.innerText = sample.label;
        btn.onclick = () => {
            document.getElementById('dialogueInput').value = sample.text;
        };
        samplesDiv.appendChild(btn);
    });
};
