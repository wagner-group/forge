# filename_mapping -- slot number -> rule stem
#
# Positional: slot NN is the NNth entry of sorted(dataset/final/*.json).
# Verify against dataset/final with:
#
#     python3 scripts/make_filename_mapping.py --check
#
# Regenerate this file with --write instead of --check.

| file | rule |
|---|---|
| `01.json` | `7zip_commandline_to_smb_share_path` |
| `02.json` | `access_lsass_memory_for_dump_creation` |
| `03.json` | `add_defaultuser_and_password_in_registry` |
| `04.json` | `add_or_set_windows_defender_exclusion` |
| `05.json` | `allow_file_and_printing_sharing_in_firewall` |
| `06.json` | `attempt_to_add_certificate_to_untrusted_store` |
| `07.json` | `bcdedit_failure_recovery_modification` |
| `08.json` | `bitsadmin_download_file` |
| `09.json` | `certutil_with_decode_argument` |
| `10.json` | `cisco_asa___logging_disabled_via_cli` |
| `11.json` | `creation_of_shadow_copy` |
| `12.json` | `deleting_shadow_copies` |
| `13.json` | `detect_html_help_spawn_child_process` |
| `14.json` | `detect_mshta_url_in_command_line` |
| `15.json` | `detect_psexec_with_accepteula_flag` |
| `16.json` | `detect_use_of_cmd_exe_to_launch_script_interpreters` |
| `17.json` | `disabling_controlpanel` |
| `18.json` | `dump_lsass_via_procdump` |
| `19.json` | `enable_wdigest_uselogoncredential_registry` |
| `20.json` | `executables_or_script_creation_in_suspicious_path` |
| `21.json` | `execution_of_file_with_multiple_extensions` |
| `22.json` | `file_download_or_read_to_pipe_execution` |
| `23.json` | `get_domainpolicy_with_powershell_script_block` |
| `24.json` | `get_domaintrust_with_powershell_script_block` |
| `25.json` | `get_domainuser_with_powershell_script_block` |
| `26.json` | `get_foresttrust_with_powershell_script_block` |
| `27.json` | `linux_auditd_dd_file_overwrite` |
| `28.json` | `linux_auditd_disable_or_modify_system_firewall` |
| `29.json` | `linux_auditd_find_ssh_private_keys` |
| `30.json` | `linux_auditd_preload_hijack_library_calls` |
| `31.json` | `linux_decode_base64_to_shell_1` |
| `32.json` | `linux_decode_base64_to_shell_2` |
| `33.json` | `linux_possible_access_to_credential_files` |
| `34.json` | `nltest_domain_trust_discovery` |
| `35.json` | `powershell_fileless_script_contains_base64_encoded_content` |
| `36.json` | `powershell_loading_dotnet_into_memory_via_reflection` |
| `37.json` | `powershell_remote_thread_injection_into_windows_system_processes` |
| `38.json` | `powershell_webrequest_using_memory_stream` |
| `39.json` | `reg_exe_manipulating_windows_services_registry_keys` |
| `40.json` | `remote_process_instantiation_via_dcom_and_powershell` |
| `41.json` | `script_execution_via_wmi` |
| `42.json` | `sdelete_application_execution` |
| `43.json` | `searchprotocolhost_network_connection` |
| `44.json` | `searchprotocolhost_no_command_line` |
| `45.json` | `serviceprincipalnames_discovery_with_powershell` |
| `46.json` | `services_lolbas_execution_process_spawn` |
| `47.json` | `set_default_powershell_execution_policy_to_unrestricted_or_bypass` |
| `48.json` | `suspicious_computer_account_name_change` |
| `49.json` | `unloading_amsi_via_reflection` |
| `50.json` | `windows_archive_collected_data_via_rar` |
| `51.json` | `windows_diskshadow_proxy_execution` |
| `52.json` | `windows_event_log_cleared` |
| `53.json` | `windows_hijack_execution_flow_version_dll_side_load` |
| `54.json` | `windows_lsa_secrets_nolmhash_registry` |
| `55.json` | `windows_new_eventlog_channelaccess_registry_value_set` |
| `56.json` | `windows_nirsoft_tool_bundle_file_created` |
| `57.json` | `windows_non_system_account_targeting_lsass` |
| `58.json` | `windows_possible_credential_dumping` |
| `59.json` | `windows_powershell_history_file_deletion` |
| `60.json` | `windows_process_injection_of_wermgr_to_known_browser` |
| `61.json` | `windows_rundll32_webdav_request` |
| `62.json` | `windows_service_creation_via_sc_exe` |
| `63.json` | `windows_sqlservr_spawning_shell` |
| `64.json` | `windows_subinacl_execution` |
| `65.json` | `windows_terminating_lsass_process` |
| `66.json` | `windows_windbg_spawning_autoit3` |
