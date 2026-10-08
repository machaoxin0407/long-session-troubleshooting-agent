"""Explain finite validator failures without inventing replacement product facts."""
GUIDANCE = {
    'procedure_checkpoint_does_not_establish_success':
        'The source places a visible event inside a procedure and assigns success '
        'to a different event. Preserve the event, action, timing and stated meaning '
        'together. Do not turn a step checkpoint into proof of normal response or '
        'successful completion. For an observation question, omit unrelated recovery '
        'instructions and retain only source-supported observations and their scope.',
    'light_observation_does_not_establish_circuit_mechanism':
        'A described light response does not itself supply the asserted circuit '
        'activation or power mechanism. Preserve the documented trigger/light '
        'behavior and the supported limit on diagnosis. Remove the unsupported '
        'circuit explanation instead of replacing it with another internal '
        'mechanism, component health claim, or invented diagnostic test.',
    'light_observation_does_not_establish_mechanical_independence':
        'The selected passage does not establish the asserted independence of '
        'the light from the motor or chuck. A limit on what a light observation '
        'can diagnose does not establish internal architecture or independent '
        'operation. Preserve documented observations and the supported inference '
        'limit without inventing a mechanism, a fault, or the opposite relationship.',
    'described_model_behavior_does_not_establish_exclusivity':
        'Describing a behavior for a named model does not establish that the '
        'behavior is unique to it or absent from every other model. Preserve '
        'the documented behavior and its conditions. Assert exclusivity only '
        'when the selected evidence explicitly states it for that same model. '
        'Do not infer the opposite claim that other models necessarily share it.',
    'red_pattern_does_not_establish_yellow_indicator_absence':
        'A description of this model\'s red flashing pattern does not establish that a '
        'yellow indicator is absent, never lights, or does not participate. Do not turn '
        'a difference in the described patterns into an exclusive hardware or behavior '
        'claim. Retain each model\'s explicitly documented pattern and conditions; '
        'state an exclusion only when its selected passage explicitly supports it.',
    'light_observation_does_not_establish_component_fault':
        'The cited light observation does not establish the asserted component fault. '
        'A failure to prove normal operation is not proof of malfunction. State only '
        'the documented observation and the limits of the inference. Do not infer '
        'normality instead, invent a diagnostic procedure, or transfer another '
        'component\'s documented fault to this component.',
    'empty_answer_requires_evidence_review':
        'No answer claims were delivered. Recheck the actual question against the '
        'provided excerpts for a supported limited answer, restriction, or referral. '
        'A question asking whether an observation establishes a conclusion can be '
        'answered by stating what the excerpts establish and the boundary of that '
        'inference; do not diagnose other components or assert what the whole manual '
        'omits. Put supported answer content in claims, not only assessment notes. '
        'Do not invent facts, procedures, prohibitions or citations to avoid an empty '
        'answer. If no relevant limited answer is supported, retain none and empty '
        'claims. This review request is not evidence that an answer exists.',
    'inference_question_conclusion_missing':
        'The question asks whether one observation establishes a conclusion, but '
        'the delivered claims only describe observations. Answer the actual '
        'inference question explicitly in a source-bound claim. Internal assessment '
        'and missing_parts are not delivered. Do not invent additional component '
        'health or a diagnostic procedure, and do not infer the opposite state '
        'merely because the requested conclusion is not established.',
    'conditional_service_restriction_broadened':
        'The cited repair restriction applies when the device encounters the '
        'listed damage or malfunction conditions. Preserve that condition on '
        'the restriction itself; a later conditional referral does not qualify '
        'an earlier unconditional prohibition. Do not invent additional fault '
        'conditions, permission to repair, or a repair procedure. Use separate '
        'claims and their actual passages for separately supported restrictions.',
    'not_included_does_not_require_purchase':
        'The source says the item is not included; this does not require the user '
        'to buy it. State the documented inclusion status without adding purchasing '
        'or installation instructions. If an operation is actually requested, '
        'derive its complete prerequisites from the relevant procedure.',
    'service_contact_method_missing_from_citation':
        'At least one selected passage does not state the contact channel asserted '
        'in this claim. Split supported service restrictions and contact details '
        'along their evidence boundaries. Do not attach all service passages to '
        'a compound claim or invent a channel, number or address.',
    'service_technician_actor_missing_from_citation':
        'At least one selected passage does not identify the asserted manufacturer '
        'technician. Preserve the exact service actor and restriction in a claim '
        'cited only to evidence that supports it; do not transfer a technician-only '
        'restriction to a passage that merely says to contact the manufacturer.',
    'selected_excerpt_does_not_establish_whole_manual_absence':
        'You have selected excerpts, not proof that the entire manual omits '
        'other information. State what the supplied excerpts establish and what '
        'they cannot establish. Do not turn missing evidence into an exhaustive '
        'claim about the whole manual, all models or all compatibility guarantees.',
    'named_model_absent_from_selected_manual_passage':
        'A factual statement names a model identifier absent from at least one '
        'selected manual passage. Recheck every selected passage separately. '
        'Do not borrow the model or its conditions from another citation, image '
        'label, prior answer or model knowledge. Split differently supported facts '
        'or omit a citation that does not support the whole retained statement. '
        'Identifier presence alone does not establish compatibility or support.',
    'storage_alias_not_established_by_each_citation':
        'The selected evidence does not independently establish the asserted card '
        'alias in every citation. A parenthesis after an alternative list can be '
        'ambiguous. Preserve the explicitly listed supported card names, capacity '
        'and inclusion status without asserting an uncertain equivalence. Do not '
        'invent an alternative alias or compatibility rule.',
    'cleaning_minimum_frequency_weakened_to_recommendation':
        'The cleaning passage gives a minimum frequency as an instruction, not '
        'merely a recommendation. Preserve the stated minimum and modality on '
        'the same cleaning action. Do not weaken it to optional advice or add '
        'stronger obligations, new frequencies or cleaning operations.',
    'light_observation_does_not_establish_component_health':
        'The cited passage describes a light response, not a diagnostic test of '
        'the circuit, trigger or switch. Do not infer component health merely '
        'because the light responds, even if you separately deny whole-device '
        'health. State the documented observation and preserve what cannot be '
        'determined. A positive health assertion needs its own explicit diagnostic '
        'support; do not invent a test procedure or replace this with a stronger '
        'whole-device claim.',
    'switch_removal_panel_caution_missing':
        'The answer gives the documented switch-puller and clip-removal method but '
        'omits an applicable surface-protection caution in that same source section. '
        'Read the cited removal passage and preserve its caution in delivered claims, '
        'with the correct object, negation and supporting citation. Assessment notes '
        'are not a delivered warning. Do not invent a new prohibition for an action '
        'that the source only calls unnecessary, add unrelated maintenance, or '
        'discard answerable steps to evade this check.',
    'unnecessary_action_misrepresented_as_prohibited':
        'A selected passage says the matched action is unnecessary; the answer '
        'turns that into a prohibition. Preserve the source modality. Unnecessary '
        'does not establish either permission or prohibition. Answer the actual '
        'question with the documented method and limits, without inventing a '
        'ban or permission. A warning about another action does not prohibit this action, '
        'but preserve that warning with its own action and object when it applies to '
        'the procedure you deliver. Correct the modality without dropping other '
        'source-supported cautions or prerequisites. '
        'Recheck every selected passage and submit the complete revised answer.',
    'supported_battery_service_limit_missing_from_empty_answer':
        'The retrieved replacement section explicitly provides a relevant service '
        'restriction and manufacturer referral. An empty answer omits this supported '
        'limit even though self-replacement steps cannot be supplied. State the '
        'documented limit using the supporting passage, preserve the unavailable '
        'procedure as missing, and use partial with insufficient=true. Do not invent '
        'disassembly steps, broaden conditional service warnings, or add contact '
        'details absent from the cited passage. This is an omission check, not '
        'verification of any previous claim.',
    'answerability_none_with_claims':
        'The submission labels the task unanswerable (none) but includes claims. '
        'Reassess the actual question and sources. If no relevant answer is '
        'supported, submit none with empty claims and insufficient=true. If a '
        'source-supported restriction or referral answers part of the requested '
        'task, state only those relevant supported facts, use partial, and '
        'preserve the missing requested procedure and insufficient=true. Do not '
        'merely relabel unrelated facts as partial or invent a prohibited procedure. '
        'The final submission must satisfy the unchanged full contract.',
    'missing_unplug_before_cleaning_prerequisite':
        'Read which component the source requires unplugging before cleaning. '
        'For an applicable cleaning instruction, preserve that prerequisite and '
        'the dependent cleaning action together in the same claim, in the documented '
        'order, citing a passage that supports both. A prerequisite in assessment '
        'notes or in another claim does not qualify this instruction. Do not extend '
        'a charger-only requirement to separately documented brush or handle rinsing, '
        'and do not add cleaning operations on components the user did not ask about.',
    'no_pump_sinkhole_not_pump_sink_destination':
        'Preserve the destination distinction in the source: no-pump models lay '
        'the hose down toward a sinkhole; pump models place it in a sink or bath. '
        'Downward direction alone does not preserve the different destination. '
        'If the destination translation is uncertain, keep the source term '
        'sinkhole instead of substituting the pump-model sink/bath destination. '
        'Do not infer an installation height or invent additional fitting steps.',
    'unnamed_indicator_does_not_identify_dirt_detect':
        'The cited operating excerpt names only an indicator, not Dirt Detect. '
        'Keep the documented action and observation without assigning a specific '
        'indicator name unless the selected evidence actually identifies it in that '
        'operating condition. A separate overview label does not establish this '
        'action-to-indicator relationship. Do not infer whole-machine health.',
    'full_bin_reinsertion_not_general_use_requirement':
        'The selected reinsertion instruction is conditional on a full bin. Preserve that '
        'condition on the action; listing a full bin as an example does not qualify a '
        'general before-reuse or after-emptying requirement. Keep the original component '
        'and do not borrow filter installation instructions to fill missing bin-refitting '
        'details. An excerpt cannot establish that the entire manual omits those details.',
    'readiness_procedure_context_missing':
        'The selected readiness excerpt belongs to a specific procedure. Preserve '
        'its reset or first-use/rinsing context in the delivered claim; generic '
        'ready wording loses that scope. Do not add related procedures to answer '
        'a brewing question. Prefer only relevant supported facts; each selected '
        'passage must support its entire claim. A scope label is not proof of support.',
    'first_use_rinsing_readiness_not_brewing_evidence':
        'The first-use passage describes heating readiness followed by rinsing. It does '
        'not independently support a compound claim that the machine is ready to brew. '
        'Check each citation separately: split facts at their evidence boundaries and '
        'use the coffee-preparation passage for brewing statements only when it actually '
        'supports them. Preserve first-use and reset scope on their own statements.',
    'drip_grid_gloss_conflicts_with_tray':
        'The cited safeguard requires two distinct components: drip tray and drip grid. '
        'An attached English label does not validate a conflicting Chinese translation. '
        'Do not label drip grid as 滴水盘, 接水盘 or 托盘. Preserve the literal source '
        'term drip grid if its translation is uncertain, and keep both components '
        'and the prohibition intact. Do not invent removal or cleaning instructions.',
    'completion_tone_is_not_visual_signal':
        'The cited completion tones are auditory. Do not call the tones a visible '
        'sign or invent a completion light. Keep the documented completion and '
        'return-to-recharge condition; distinguish any separately supported visual observation.',
    'reboot_action_occurs_before_success':
        'The cited hold-and-release sequence performs the reboot; it does not '
        'occur after reboot success. Preserve the order: hold CLEAN for 10 seconds '
        'until indicators illuminate, then release; the release tone signifies '
        'success. Do not add this procedure to an unrelated state question.',
    'reboot_hold_indicators_are_not_success_state':
        'Preserve the sequence in the cited manual: all indicators illuminate while '
        'holding CLEAN before release; the tone upon release signifies successful reboot. '
        'Do not assert that all indicators illuminate after successful reboot. For a '
        'state-only question, report only the documented observation and its trigger, '
        'without adding a control tutorial or inferring general normality.',
    'vacuum_control_tutorial_does_not_answer_state_question':
        'The question asks for state indicators, not control instructions. Keep only '
        'source-supported observations with their actual state and trigger conditions. '
        'Do not append reboot, start, pause, resume or standby button procedures to '
        'make an incomplete state answer longer. A cited procedure is not evidence '
        'that the device is currently normal. Preserve unresolved state limitations.',
    'complete_answer_omits_oily_smoke_handling':
        'The question also asks what to do about oily or non-dark smoke. The delivered '
        'dark-smoke response and facts about smoke occurring do not answer that part. '
        'Check the actual sources for applicable handling guidance. If it is absent, '
        'explicitly preserve the unresolved oily-smoke part and assess partial. Do not '
        'normalize fatty-food smoke or invent cleaning, stopping or continued-use steps '
        'to obtain a complete label. Any added action must independently pass source checks.',
    'routine_cleaning_not_smoke_response':
        'The cited passage describes routine cleaning, not a response to smoke. '
        'Do not present its removal, cooling, degreasing or brushing steps as smoke '
        'treatment without a source explicitly connecting them to that condition. '
        'Keep the supported smoke warning with its own conditions and cite each '
        'fact to its actual passage. If ordinary oily-smoke handling is undocumented, '
        'retain that limitation rather than substituting an adjacent cleaning procedure.',
    'complete_answer_omits_requested_washer_start':
        'The user asks how to start the washer, but the delivered answer only covers '
        'preparation and settings. These do not establish the start transition. Re-read '
        'the actual evidence for a supported start action. If no such transition is '
        'documented, retain the supported settings, explicitly identify the unresolved '
        'start mechanism, and assess partial rather than complete. Do not invent a '
        'START button or automatic execution merely to satisfy this coverage check.',
    'washing_temperature_precaution_missing':
        'The current washing manual explicitly prohibits excessively hot water, including '
        '50 degrees Celsius or more. Preserve the temperature precaution, inclusive boundary '
        'and unit in the delivered washing procedure with its actual source. Mentioning it '
        'only in assessment notes does not preserve the precaution for the user.',
    'washing_pressure_precaution_missing':
        'The current washing manual says to close the water tap a little if water pressure '
        'is too high. Preserve that conditional partial closure in the delivered procedure '
        'with its actual source. A generic instruction to adjust the tap omits the direction; '
        'assessment notes are not a substitute for the delivered precaution.',
    'inner_cover_is_not_laundry_container':
        'The source says to cover the laundry with the inner cover before spinning. '
        'The cover goes over the laundry; it is not a container into which laundry '
        'is inserted. Preserve the object relationship as well as the before-spinning condition.',
    'inner_cover_relation_reversed':
        'The selected manual requires the inner cover over the laundry before spinning. '
        'Do not reverse the relationship into laundry covering the inner cover. '
        'Preserve which object covers which, and the before-spinning condition, in the actual claim.',
    'absence_of_error_signal_not_established':
        'The selected passage describes the distress sound, spoken message and '
        'blinking troubleshooting indicator that report a problem. It does not '
        'provide an exhaustive diagnostic rule for their absence. Retain the '
        'documented positive signals without concluding that no error is reported '
        'or that the machine is normal when those signals are absent.',
    'reboot_hold_requires_ten_seconds':
        'The selected reboot procedure says to press and hold CLEAN for 10 seconds '
        'until all indicators illuminate, then release. If describing that action, '
        'preserve its duration, indicator condition and order. A tone describing '
        'successful reboot is not a substitute for a complete reboot instruction.',
    'reboot_tone_requires_button_release':
        'The cited procedure says to hold CLEAN for 10 seconds until all indicators '
        'illuminate, then release. The audible reboot tone occurs when the button '
        'is released. Preserve the release step and timing when describing this '
        'action and its tone; holding alone is not the complete sequence.',
    'inner_cover_requires_spinning_scope':
        'The cited manual requires the inner cover on laundry before spinning. '
        'Preserve that operation and timing in the delivered sentence. Do not turn '
        'it into a before-washing or unconditional instruction. The wash tub lid '
        'is a separate component and precaution; it does not widen the inner-cover rule.',
    'washing_timer_step_missing':
        'The selected To Wash procedure explicitly sets WASH TIMER to 1-15 minutes. '
        'Retain this setting and its exact range and unit in the delivered claims '
        'with the supporting passage, not only in assessment notes. This setting '
        'does not by itself establish automatic starting or normal machine operation.',
    'washing_water_detergent_step_missing':
        'The current To Wash procedure explicitly includes filling water and adding '
        'detergent before setting the timer. Retain both actions in delivered claims '
        'with their actual passage, not just assessment notes. Do not skip material '
        'preparation while adding unrelated control descriptions.',
    'washing_laundry_level_step_missing':
        'The current To Wash procedure explicitly includes loading laundry and filling '
        'water to the H high water level before setting the timer. Preserve both '
        'actions and their source order. H limits the water level, not the laundry load.',
    'tray_safeguard_requires_distinct_grid':
        'The cited safeguard names both drip tray and drip grid. They are distinct '
        'components. Preserve both in the safety condition; do not translate grid '
        'as a second tray or omit it. Retain drip grid in the answer if the translation '
        'is uncertain. Do not invent removal or assembly instructions.',
    'heating_duration_not_in_cited_passage':
        'The claim states a 25-second heating duration, but one of its selected '
        'espresso-machine passages does not document that duration. Each citation '
        'must support the whole claim. Split the timed heating fact from reset '
        'readiness, and bind each to its actual source; do not borrow the duration '
        'from another passage or add it to the reset procedure.',
    'pan_smoke_warning_does_not_support_basket_removal':
        'The selected smoke warning names pan as the component to pull out after smoke stops. '
        'Do not substitute basket or 炸篮, and do not label a basket as pan in parentheses. '
        'Preserve the source component identity; retain the original word pan if its translation '
        'would be ambiguous. Keep the dark-smoke condition, immediate unplugging and waiting '
        'until smoke stops attached to the same sourced instruction. Do not borrow a basket '
        'instruction from a different steam warning to rewrite this smoke warning.',
    'complete_answer_omits_requested_tray_removal':
        'The question asks how to remove the drip tray, but the delivered claims only '
        'describe adjacent tasks or safety conditions. Emptying the tray is not a removal '
        'procedure. Re-read the evidence for that requested action; do not invent it. If '
        'it is not supported, mark the missing removal part explicitly and use a partial '
        'or none assessment rather than claiming the whole task is complete.',
    'cartridge_readiness_not_in_cited_video':
        'This readiness narration is attached to an additional video passage that does not '
        'contain the cited fault-LED narration. A different control-panel video is not '
        'additional support for that transcript. Split distinct facts and retain only '
        'the passage actually supporting each, with its tutorial applicability.',
    'washing_lid_precaution_missing_from_answer':
        'The current washing manual explicitly requires closing the wash tub lid to avoid splashes. '
        'Preserve that precaution in the delivered procedure with its actual supporting passage. '
        'Name the wash tub lid explicitly; generic washer lid or 洗衣机盖 leaves the component ambiguous. '
        'Retain wash tub lid in the sentence if the translation is uncertain. '
        'Do not substitute the spin basket lid or inner spin cover. Unnumbered precautions still '
        'apply alongside numbered operating steps; group facts sharing the same source when needed.',
    'first_use_smoke_normality_not_in_citation':
        'The cited passage does not establish normal first-use smoke. Bind the claim '
        'to the passage that actually states both that condition and normality, or omit it. '
        'A dark-smoke warning does not support this exception. Do not duplicate the observation.',
    'fatty_food_smoke_not_established_normal':
        'The cited passage says fatty food can emit smoke but does not call it normal. '
        'Do not transfer the first-use normality exception to fatty food or other smoke. '
        'Preserve the documented condition and avoid adding reassurance not present in evidence.',
    'cartridge_video_scope_missing':
        'This readiness statement cites a video categorized as a cartridge-related source. '
        'Keep that video-specific applicability explicit in the claim, and attribute what '
        'its supplied transcript actually says. The topic/title is provenance, not proof '
        'of a repair, a model identity, or general printer readiness. Do not infer extra events.',
    'fault_indicator_identity_specialized':
        'The cited narration names a printer-fault indicator, not a cartridge-fault indicator. '
        'Keep the transcript\'s indicator identity unchanged. Attribute the statement to the '
        'cartridge-related video without turning its topic into a component identity. '
        'Source metadata restricts applicability; it does not supply additional device facts.',
    'spin_basket_warning_does_not_identify_drum_motion':
        'The cited safety warning names the spin dryer basket and spinning laundry. '
        'It does not identify a drum or prove a current observed machine state. '
        'Preserve the named components and warning context; do not use another '
        'model\'s video observation to reinterpret this manual. Cite observations '
        'separately with their own device scope, or omit the unsupported inference.',
    'washing_pocket_prerequisite_missing_from_answer':
        'The current manual explicitly gives a before-washing pocket precaution. '
        'A washing procedure must retain this prerequisite in delivered claims with '
        'its actual supporting passage, not only in assessment notes. Read the current '
        'catalog for that passage and preserve its timing. Group steps sharing the same '
        'evidence when needed so the claim limit does not remove necessary precautions.',
    'water_selector_does_not_establish_level_control':
        'Keep the literal WATER SELECTOR control label. The selected passage does not '
        'identify it as a water-level selector; do not add an unsupported control function in translation.',
    'hot_water_limit_excludes_equality':
        'The selected warning prohibits water at 50°C or higher, including exactly 50°C. '
        'Preserve this exclusive upper limit; do not replace it with at most 50°C. '
        'Resubmit the complete claim with accurate source binding.',
    'full_bin_completion_requires_default_setting':
        'The source describes cycle completion on a full bin as a default setting that can be changed. Preserve the default-setting condition on that behavior, and do not assert it unconditionally or regardless of user settings. Separate it from the rule about starting with a full bin.',
    'unfinished_citation_scaffold':
        'A claim ends in an unfinished citation scaffold. Submit a complete plain-text claim and bind citations only through the structured passage IDs. Re-check its whole meaning and evidence; do not simply trim the damaged suffix.',
    'waking_feedback_is_not_cleaning_start':
        'The source distinguishes waking the device from starting a cleaning cycle. Bind each button action and its feedback to its stated state; a wake indication does not establish that cleaning has started.',
    'timer_setting_does_not_establish_start_trigger':
        'The cited text documents a timer setting, not an explicit start trigger. Preserve the documented setting without adding an unsupported automatic-start effect; identify the missing start detail if needed.',
    'simulated_source_cannot_establish_actual_operation':
        'The cited sources explicitly describe simulated imagery. Preserve that scope; it cannot establish the current operating state of a real device. Describe only supported demonstration content or the remaining uncertainty.',
    'whole_system_emptying_is_not_tray_removal':
        'The linked procedure performs a different task. A final mention of the requested component does not make preceding steps instructions for removing it. Use task-matching evidence; identify any unsupported subtask as missing.',
    'reinsertion_not_in_selected_manual_quote':
        'The cited passage does not describe the claimed reinsertion operation. Preserve supported parts and explicitly identify the missing subtask; do not invent assembly or locking actions.',
    'video_summary_or_ocr_is_not_instruction_transcript':
        'Pooled OCR and visual summaries do not establish a complete ordered procedure. Do not turn scattered labels into commands or fill in absent steps. Use a source that actually documents the requested procedure, or identify its absence.',
    'motion_does_not_establish_spin_cycle':
        'Observed motion does not establish a named cycle. Do not substitute ordinary motion or manual setup/safety text for evidence of the current state.',
    'timer_setup_is_not_current_cycle_observation':
        'Instructions to set a timer do not show which cycle is currently running. Distinguish documented setup from actual state observations.',
    'lid_interlock_is_not_current_cycle_observation':
        'A description of what the brake does when a lid is opened is a conditional '
        'safety mechanism, not an observation identifying the currently running cycle. '
        'Do not offer opening the lid as a diagnostic action or repeat the interlock '
        'description as the answer to the current-state question. Use evidence that '
        'actually identifies the requested state, keeping each device and cycle scope '
        'separate. If the evidence only shows unspecified motion, preserve that limit '
        'rather than labeling it as the requested cycle.',
    'safety_instructions_do_not_establish_device_health':
        'Safety requirements describe how to use a device, not evidence that it is currently functioning normally. Do not convert allowed conditions or prohibitions into a health test; distinguish source requirements from actual state observations.',
    'component_observation_cannot_establish_device_health':
        'A component indicator, display or local observation does not prove overall device health. State only supported observations and distinguish what remains unknown; do not relabel partial evidence as normal operation.',
    'missing_drain_pump_qualification':
        'Bind each mode to its actual pump/no-pump condition in the source. Merely mentioning a pump is insufficient; do not reverse the alternatives or omit an applicable branch.',
    'missing_b_type_qualification':
        'A cited action is limited to a subtype. Preserve that limitation on the action rather than applying it to every device.',
    'ambiguous_drain_hose_branch':
        'The source gives different drain-hose arrangements for pump and no-pump models. Saying the arrangement depends on the model without specifying which arrangement applies does not preserve those instructions. State the distinct supported branches or identify the unresolved model. Within each branch, preserve both the direction and destination of the hose action exactly as documented. Naming a drain destination does not preserve a separately stated downward direction. A direction in another model branch cannot supply the missing qualifier.',
    'partial_tap_closure_condition_lost':
        'Preserve both the existing-pressure condition and the limited extent of the tap adjustment. Do not turn a conditional slight adjustment into full closure or a preventative action.',
    'surface_cleaning_does_not_establish_tray_method':
        'The cited method names the appliance surface. Do not transfer that method to a named removable component without explicit support. Preserve the actual cleaning object and state missing component instructions when needed.',
    'claim_markup_requires_rewrite':
        'Rewrite a single prose block without display tags, embedded source markers or line breaks. Select references only through passage_ids and images only through image_ids. Recheck all factual claims against the supplied sources; fixing formatting does not establish support.',
    'fatty_food_smoke_not_in_cited_passage':
        'The selected passage does not state the asserted relation between fatty food and smoke. Use only a passage that actually supports that relation, or omit the assertion; a dark-smoke response warning is not evidence of a smoke cause.',
    'food_doneness_smoke_cause_not_in_cited_passage':
        'Do not invent a relationship between food doneness and smoke or add a cooking instruction to prevent it. Re-read the cited source and retain only its supported causes and actions.',
    'filter_door_condition_not_in_selected_quote':
        'The selected citation does not state the asserted filter-door closing condition. '
        'A passage about emptying or reinserting the bin, or merely cleaning the filter, '
        'cannot support that condition. Re-read the catalog and cite the actual supporting '
        'passage for a separate relevant claim, or omit the unsupported clause. '
        'Do not infer a required order between filter installation and bin refitting.',
    'procedure_requires_revalidation':
        'At least one instruction in the candidate answer failed validation, so the remaining '
        'claims require joint revalidation. This flag does not establish that this claim is '
        'false or that a prerequisite dependency exists. Rebuild a coherent answer to the '
        'requested task using the original sources. Do not turn a state-identification '
        'question into operating instructions. When the requested task is a procedure, '
        'preserve its actual prerequisites and dependent steps; do not silently drop a '
        'prerequisite while retaining dependent actions. Recheck each claim and citation '
        'rather than treating this flag as additional product evidence.',
    'selected_image_outside_cited_passages':
        'This known image is not in the passages cited by the answer. Choose only useful images from actually cited passages, or choose no image. Do not add irrelevant citations solely to justify an image.',
}


def guidance_for(reason):
    return GUIDANCE.get(reason,
        'Re-read the linked complete sources and correct this scope mismatch. Preserve conditions, entities and task scope; the failure label itself is not evidence.')
