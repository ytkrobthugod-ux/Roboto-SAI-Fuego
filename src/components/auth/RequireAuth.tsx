/**
 * RequireAuth - Guards protected routes
 * Redirects to /login if not authenticated
 */

import { useEffect, useState, type ReactNode } from 'react';
import { useNavigate, Outlet } from 'react-router-dom';
import { useAuthStore } from '@/stores/authStore';

interface RequireAuthProps {
  children?: ReactNode;
}

export const RequireAuth = ({ children }: RequireAuthProps) => {
  const { isLoggedIn, refreshSession } = useAuthStore();
  const navigate = useNavigate();
  const [isChecking, setIsChecking] = useState(!isLoggedIn);

  useEffect(() => {
    // If we believe we are logged in, we render the content.
    // The App.tsx global listener handles background validation.
    if (isLoggedIn) {
      setIsChecking(false);
      return;
    }

    // If we are NOT logged in, we must verify with the backend
    // before redirecting (in case of page refresh where state is lost but cookie exists)
    let active = true;
    const verify = async () => {
      // Only verify if we really don't know (isChecking is true)
      // If we are just "logged out", this might be redundant but safe.
      const ok = await refreshSession();
      if (!active) return;
      
      if (!ok) {
        navigate('/login', { replace: true });
      } else {
        // Validation succeeded, isLoggedIn should be true now (via store update)
        setIsChecking(false);
      }
    };

    void verify();

    return () => {
      active = false;
    };
  }, [isLoggedIn, refreshSession, navigate]);

  // While checking, show nothing (or loading spinner)
  if (isChecking) return null;

  // If check finished and we are still not logged in, we are redirecting anyway.
  if (!isLoggedIn) return null;
  
  // Support both direct children (for testing) and Outlet (for nested routes)
  return children ? <>{children}</> : <Outlet />;
};
